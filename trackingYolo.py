#!/usr/bin/env python3

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import rclpy
import random
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import math
from ultralytics import YOLO

class SimpleObjectTracker:
    """Sistema simple de tracking para estabilizar detecciones"""
    
    def __init__(self, max_distance=100, max_time_missing=0.5):
        self.tracked_objects = []  # Lista de objetos tracked
        self.next_id = 0
        self.max_distance = max_distance  # Distancia máxima en píxeles para considerar el mismo objeto
        self.max_time_missing = max_time_missing  # Tiempo máximo sin detectar (en segundos)
        
    def update(self, detections, current_time):
        """Actualiza el tracker con nuevas detecciones"""
        # Marcar todos los objetos como no vistos en este frame
        for obj in self.tracked_objects:
            obj['seen_this_frame'] = False

        # Para cada detección, buscar coincidencia con objeto existente
        for detection in detections:
            center_x = detection['center_x']
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bbox']

            best_match = None
            best_distance = float('inf')

            # Buscar el objeto más cercano de la misma clase
            for obj in self.tracked_objects:
                if obj['label'] == label and obj['active']:
                    distance = abs(center_x - obj['center_x'])
                    
                    if distance < self.max_distance and distance < best_distance:
                        best_distance = distance
                        best_match = obj

            if best_match is not None:
                # Actualizar objeto existente con suavizado
                alpha = 0.6  # Factor de suavizado (0.6 = 60% nuevo, 40% anterior)
                best_match['center_x'] = int(alpha * center_x + (1 - alpha) * best_match['center_x'])
                best_match['confidence'] = max(best_match['confidence'], confidence)
                best_match['bbox'] = bbox
                best_match['last_update'] = current_time
                best_match['last_seen'] = current_time
                best_match['seen_this_frame'] = True
                best_match['detection_count'] += 1
                best_match['stable_count'] += 1
            else:
                # Crear nuevo objeto
                new_obj = {
                    'id': self.next_id,
                    'label': label,
                    'center_x': center_x,
                    'confidence': confidence,
                    'bbox': bbox,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_update': current_time,
                    'active': True,
                    'seen_this_frame': True,
                    'detection_count': 1,
                    'stable_count': 0  # Contador de detecciones estables
                }
                self.tracked_objects.append(new_obj)
                self.next_id += 1

        # Desactivar objetos que no fueron vistos recientemente
        for obj in self.tracked_objects:
            if not obj['seen_this_frame']:
                if current_time - obj['last_seen'] > self.max_time_missing:
                    obj['active'] = False
            else:
                # Reactivar si estaba inactivo
                obj['active'] = True
    
    def get_best_detection(self):
        """Retorna la mejor detección (más estable y con mayor confianza)"""
        active_objects = [obj for obj in self.tracked_objects if obj['active'] and obj['stable_count'] > 1]
        
        if not active_objects:
            return None
        
        # Ordenar por estabilidad y confianza
        best_obj = max(active_objects, key=lambda x: (x['stable_count'], x['confidence']))
        
        return {
            'center_x': best_obj['center_x'],
            'confidence': best_obj['confidence'],
            'bbox': best_obj['bbox'],
            'label': best_obj['label'],
            'stable_count': best_obj['stable_count']
        }
    
    def get_all_active(self):
        """Retorna todos los objetos activos"""
        return [obj for obj in self.tracked_objects if obj['active']]

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        
        self.bridge = CvBridge()
        
        # Modelo YOLOv8
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ Modelo YOLOv8 cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return
        
        # Sistema de tracking
        self.tracker = SimpleObjectTracker(max_distance=200, max_time_missing=1.5)
        
        # Clase objetivo
        self.target_class = None
        self.class_selected = False
        self.request_target_class()
        
        # Control de movimiento
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Variables de estado
        self.object_detected = False
        self.object_center_x = None
        self.last_object_center_x = None
        self.image_center_x = None
        
        # Variables de distancia LIDAR
        self.front_distance = float('inf')
        self.front_value_percentil = float('inf')
        self.left_value = float('inf')
        self.right_value = float('inf')
        
        # Parámetros de tracking mejorados
        self.CENTERING_THRESHOLD = 200   # Reducido para ser más estricto
        self.MIN_DISTANCE = 1.5
        
        # Estados de búsqueda
        self.search_start_time = None
        self.SEARCH_DURATION = 20.0
        self.search_direction = 0.15
        self.is_searching = False
        self.is_autonomous_nav = False
        self.is_reversing = False
        self.reverse_timer = 0
        self.turning_direction = None
        
        # Variable para la ventana OpenCV
        self.latest_frame = None
        
        # CAMBIOS PARA CORTES DE CÁMARA:
        self.consecutive_image_errors = 0  # Contador de errores consecutivos de imagen
        self.last_successful_image_time = time.time()  # Último timestamp de imagen exitosa
        
        # ✅ NUEVAS VARIABLES PARA CONTROL DE FRAME RATE
        self.frame_counter = 0
        self.process_every_n_frames = 3  # Procesar solo 1 de cada 3 frames
        
        # Suscripciones
        lidar_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile=lidar_qos)
        
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, sensor_qos)
        
        # ✅ TIMER MÁS LENTO para dar tiempo al procesamiento
        self.create_timer(0.15, self.control_loop)  # Cambio: 0.1 -> 0.15
        
        self.get_logger().info("Object Tracker mejorado iniciado. Esperando selección de clase...")

    def request_target_class(self):
        """Solicitar clase objetivo mediante ventana tkinter"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
            
            user_input = simpledialog.askstring(
                "Object Tracker - YOLOv8 Mejorado",
                "Ingrese la clase a detectar (ej: 'person', 'car', 'bottle'):",
                parent=root
            )
            
            if user_input and user_input.strip():
                self.target_class = user_input.strip().lower()
                messagebox.showinfo("Configuración", f"✅ Detectando y siguiendo: {self.target_class}")
            else:
                self.target_class = 'person'
                messagebox.showinfo("Configuración", "Usando clase por defecto: person")
            
            self.class_selected = True
            root.destroy()
            print(f"✅ Detectando: {self.target_class}")
            
        except Exception as e:
            self.get_logger().error(f"Error con ventana tkinter: {e}")
            print("\n" + "="*50)
            print("OBJECT TRACKER MEJORADO - YOLOv8")
            print("="*50)
            print("No se pudo abrir ventana gráfica. Usando entrada por terminal:")
            target = input("Ingrese la clase a detectar (ej: 'person', 'car') [default: person]: ").strip()
            
            self.target_class = target.lower() if target else 'person'
            self.class_selected = True
            print(f"✅ Detectando: {self.target_class}")
            print("="*50 + "\n")

    # Función para obtener el percentil del rango de detecciones
    def get_sector_distance(self, ranges, indices):
        if len(indices) == 0:
            return 20.0
        return np.percentile(ranges[indices], 20)
    
    def laser_callback(self, msg):
        """Obtener distancias del LIDAR - CORREGIDO"""
        ranges = np.array(msg.ranges)
        
        # CORREGIDO: Mismo procesamiento robusto que obstacle_avoider
        ranges = np.where(np.isfinite(ranges), ranges, 20.0)
        ranges = np.clip(ranges, 0.0, 20.0)  # Limitar valores máximos
        
        # Sector frontal (±15°)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        front_indices = np.where((angles > -0.26) & (angles < 0.26))[0]
        
        # Para tracking: usar mínimo (más sensible a obstáculos cercanos)
        if len(front_indices) > 0:
            self.front_distance = np.min(ranges[front_indices])
        else:
            self.front_distance = 20.0

        # Para navegación autónoma: usar percentil (más robusto a ruido)
        self.front_value_percentil = self.get_sector_distance(ranges, front_indices)

        # Derecha: -π a -0.26
        right_indices = np.where((angles >= -np.pi) & (angles <= -0.26))[0]
        self.right_value = self.get_sector_distance(ranges, right_indices)

        # Izquierda: +0.26 a +π
        left_indices = np.where((angles >= 0.26) & (angles <= np.pi))[0]
        self.left_value = self.get_sector_distance(ranges, left_indices)

    def detect_objects(self, frame):
        """Detectar objetos usando YOLOv8 con parámetros optimizados"""
        try:
            # Parámetros optimizados para mejor estabilidad
            results = self.model(frame, verbose=False, conf=0.6, iou=0.5)  
            
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id].lower()
                    confidence = float(box.conf[0])
                    
                    if cls_name == self.target_class:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = (x1 + x2) // 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center_x': center_x,
                            'confidence': confidence,
                            'label': cls_name
                        })
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"Error en detección: {e}")
            return []

    def image_callback(self, msg):
        """Procesar imagen y detectar objetos con tracking"""
        if not self.class_selected:
            return
        
        # ✅ CONTROL DE FRAME RATE - Skip frames
        self.frame_counter += 1
        if self.frame_counter < self.process_every_n_frames:
            return  # Skip este frame
        self.frame_counter = 0  # Resetear contador
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_h, frame_w = frame.shape[:2]
            self.image_center_x = frame_w // 2
            
            # SOLO CAMBIO: Marcar imagen exitosa
            self.consecutive_image_errors = 0
            self.last_successful_image_time = time.time()
            
            # Detectar objetos con YOLO
            raw_detections = self.detect_objects(frame)
            
            # Actualizar tracker
            current_time = time.time()
            self.tracker.update(raw_detections, current_time)
            
            # Obtener la mejor detección estable
            best_detection = self.tracker.get_best_detection()
            
            if best_detection is not None:
                self.object_detected = True
                self.last_object_center_x = self.object_center_x
                self.object_center_x = best_detection['center_x']
                
                # Resetear estados de búsqueda
                self.is_searching = False
                self.is_autonomous_nav = False
                self.search_start_time = None
                
                print(f"✅ Objeto ESTABLE: {best_detection['label']} (conf: {best_detection['confidence']:.2f}, estabilidad: {best_detection['stable_count']}) en x={self.object_center_x}")
            else:
                if self.object_detected:
                    # Determinar dirección de búsqueda basada en última posición conocida
                    if self.object_center_x is not None and self.image_center_x is not None:
                        if self.object_center_x > self.image_center_x:
                            print("🔍 Objeto perdido - última pos: DERECHA, buscando a la derecha")
                            self.search_direction = -0.3
                        else:
                            print("🔍 Objeto perdido - última pos: IZQUIERDA, buscando a la izquierda")
                            self.search_direction = 0.3
                    else:
                        self.search_direction = 0.3
                
                self.object_detected = False
                self.object_center_x = None
                
                if not self.is_searching and not self.is_autonomous_nav:
                    self.is_searching = True
                    self.search_start_time = time.time()
                    print("🔄 Iniciando búsqueda giratoria por 20 segundos")
            
            # VISUALIZACIÓN MEJORADA
            annotated_frame = frame.copy()
            
            # Mostrar todas las detecciones raw (en gris)
            for det in raw_detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(annotated_frame, f"raw {det['confidence']:.2f}", (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Mostrar objetos tracked (en colores)
            all_active = self.tracker.get_all_active()
            for obj in all_active:
                x1, y1, x2, y2 = obj['bbox']
                
                # Color para las detecciones
                color = (0, 255, 0) # Color verde
                
                
                thickness = 3 if best_detection and obj['id'] == best_detection.get('id', -1) else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Información del tracking
                label_text = f"ID:{obj['id']} {obj['label']} {obj['confidence']:.2f}"
                stable_text = f"Stable:{obj['stable_count']}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(annotated_frame, stable_text, (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Centro del objeto tracked
                center_x = obj['center_x']
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 8, color, -1)
            
            # Líneas de referencia
            cv2.line(annotated_frame, (self.image_center_x, 0), (self.image_center_x, frame_h), (0, 255, 0), 2)
            
            # Zona de tolerancia
            left_bound = self.image_center_x - self.CENTERING_THRESHOLD
            right_bound = self.image_center_x + self.CENTERING_THRESHOLD
            cv2.line(annotated_frame, (left_bound, 0), (left_bound, frame_h), (0, 255, 255), 1)
            cv2.line(annotated_frame, (right_bound, 0), (right_bound, frame_h), (0, 255, 255), 1)
            
            # Título mejorado con información de tracking
            if self.object_detected and best_detection:
                if self.front_distance <= self.MIN_DISTANCE:
                    title = f"PARADO - Objetivo alcanzado: {self.target_class} (Dist: {self.front_distance:.1f}m)"
                    color = (0, 0, 255)
                else:
                    title = f"TRACKING ESTABLE: {self.target_class} (Estab: {best_detection['stable_count']}, Dist: {self.front_distance:.1f}m)"
                    color = (0, 255, 0)
            elif self.is_searching:
                elapsed = time.time() - self.search_start_time if self.search_start_time else 0
                title = f"BUSCANDO: {self.target_class} ({elapsed:.1f}/{self.SEARCH_DURATION}s)"
                color = (0, 165, 255)
            elif self.is_autonomous_nav:
                title = f"NAVEGACION AUTONOMA - Buscando: {self.target_class}"
                color = (255, 0, 0)
            else:
                title = f"BUSQUEDA INICIAL - Objetivo: {self.target_class}"
                color = (128, 0, 128)
            
            cv2.putText(annotated_frame, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Información de estado mejorada
            total_raw = len(raw_detections)
            total_tracked = len(all_active)
            
            if self.object_detected and best_detection:
                error = abs(self.object_center_x - self.image_center_x)
                centered = error <= self.CENTERING_THRESHOLD
                status_text = f"Raw: {total_raw}, Tracked: {total_tracked}, Error: {error}px, Centrado: {centered}"
                status_color = (0, 255, 0)
            else:
                status_text = f"Raw detections: {total_raw}, Tracked objects: {total_tracked}"
                status_color = (128, 0, 128)
            
            cv2.putText(annotated_frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            self.latest_frame = annotated_frame
                
        except Exception as e:
            # SOLO CAMBIO: Manejar errores de imagen de forma más robusta
            self.consecutive_image_errors += 1
            self.get_logger().error(f"Error procesando imagen (#{self.consecutive_image_errors}): {e}")
            
            # Si hay demasiados errores, mostrar mensaje pero seguir intentando
            if self.consecutive_image_errors > 10:
                self.get_logger().warn("Demasiados errores de imagen consecutivos, pero continuando...")
                self.consecutive_image_errors = 0  # Resetear para seguir intentando

    def control_loop(self):
        """Loop principal de control del robot - MEJORADO"""
        if not self.class_selected:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return
            
        cmd = Twist()

        self.is_searching = False
        self.is_autonomous_nav = True

        # TRACKING con objeto detectado de manera estable
        if self.object_detected and self.object_center_x is not None:
            error_x = self.object_center_x - self.image_center_x
            is_centered = abs(error_x) <= self.CENTERING_THRESHOLD
            
            print(f"🎯 TRACKING ESTABLE - Dist: {self.front_distance:.2f}m, Error: {error_x}px, Centrado: {is_centered}")
            
            if self.front_distance <= self.MIN_DISTANCE:
                # MUY CERCA - Solo centrar sin avanzar
                cmd.linear.x = 0.0
                
                if not is_centered:
                    # Giro más suave cuando está cerca
                    angular_speed = 0.10 
                    if error_x > 0:
                        cmd.angular.z = -angular_speed
                    else:
                        cmd.angular.z = angular_speed
                    print(f"🛑 PARADO - Centrando suavemente (error: {error_x}px)")
                else:
                    cmd.angular.z = 0.0
                    print(f"✅ OBJETIVO ALCANZADO! Dist: {self.front_distance:.2f}m")
            else:
                # DISTANCIA SEGURA - Avanzar y centrar
                if not is_centered:
                    # Centrar mientras avanza lentamente
                    angular_speed = 0.20 if abs(error_x) > 300 else 0.15  # Adaptativo
                    if error_x > 0:
                        cmd.angular.z = -angular_speed
                    else:
                        cmd.angular.z = angular_speed
                    
                    cmd.linear.x = 0.2  # Muy lento mientras centra
                    print(f"🎯 Centrando - error: {error_x}px (velocidad angular: {angular_speed})")
                else:
                    # CENTRADO - Avanzar directamente
                    cmd.linear.x = 0.25
                    cmd.angular.z = -0.05
                    print(f"✅ Centrado - avanzando directamente")
        
        # Búsqueda giratoria
        elif self.is_searching:
            current_time = time.time()
            elapsed_time = current_time - self.search_start_time
            
            if elapsed_time < self.SEARCH_DURATION:
                cmd.linear.x = 0.0
                cmd.angular.z = self.search_direction
                print(f"🔄 Búsqueda giratoria - {elapsed_time:.1f}/{self.SEARCH_DURATION}s")
            else:
                self.is_searching = False
                self.is_autonomous_nav = True
                print("🗺️ Búsqueda terminada - navegación autónoma")
        
        # Navegación autónoma (igual que antes)
        elif self.is_autonomous_nav:
            print(f"🗺️ NAVEGACIÓN: Front: {self.front_value_percentil:.2f}, Left: {self.left_value:.2f}, Right: {self.right_value:.2f}")

            # Si está en modo de retroceso
            if self.is_reversing:
                cmd.linear.x = -0.2
                self.reverse_timer += 1

                # Después de un tiempo, aproximadamente 2 segundos
                if self.reverse_timer > 20:
                    self.is_reversing = False
                    self.reverse_timer = 0
                    # Girar aleatoriamente después de retroceder
                    self.turning_direction = random.choice([-0.2, 0.2])

            # Si está en modo de giro
            elif self.turning_direction is not None:
                if self.front_value_percentil > 0.8:
                    self.turning_direction = None # Dejar de girar cuando la distancia sea segura
                else:
                    cmd.angular.z = self.turning_direction # Continuar girando en la misma dirección

            # Movimiento normal cuando hay espacio suficiente adelante
            elif self.front_value_percentil > 1.5:
                cmd.linear.x = 0.25
                if self.left_value < 0.4:
                    cmd.angular.z = -0.2
                elif self.right_value < 0.4:
                    cmd.angular.z = 0.2
                else:
                    cmd.angular.z = -0.05

            # Movimiento más lento cuando se acerca a obstáculos
            elif self.front_value_percentil >= 0.9:
                cmd.linear.x = 0.15
                if self.left_value < 0.4:
                    cmd.angular.z = -0.2
                elif self.right_value < 0.4:
                    cmd.angular.z = 0.2
                else:
                    cmd.angular.z = -0.05

            # Encuentra un obstáculo cercano
            else:
                cmd.linear.x = 0.0 # Detener el movimiento

                # Comprobar si hay obstáculos en ambos lados
                left_blocked = self.left_value < 0.5
                right_blocked = self.right_value < 0.5

                # Si hay obstáculos en ambos lados, retroceder
                if left_blocked and right_blocked:
                    self.is_reversing = True
                    cmd.linear.x = -0.15
                    self.turning_direction = None
                # Si hay obstáculos a la izquierda, girar a la derecha
                elif left_blocked:
                    self.turning_direction = -0.2
                    cmd.angular.z = self.turning_direction
                # Si hay obstáculos a la derecha, girar a la izquierda
                elif right_blocked:
                    self.turning_direction = 0.2
                    cmd.angular.z = self.turning_direction
                # Si ambos lados están libres, elegir aleatoriamente una dirección
                else:
                    self.turning_direction = random.choice([-0.2, 0.2])
                    cmd.angular.z = self.turning_direction

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTracker()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            if node.latest_frame is not None:
                cv2.imshow("Object Tracker - YOLOv8", node.latest_frame)
            else:
                # SOLO CAMBIO: Verificar si hay problemas con la cámara
                time_since_last_image = time.time() - node.last_successful_image_time
                if time_since_last_image > 2.0:  # Si no hay imagen por más de 2 segundos
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, f"Esperando imagen... Sin señal por {time_since_last_image:.1f}s", 
                               (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(blank_frame, f"Objetivo: {node.target_class if node.target_class else 'None'}", 
                               (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Object Tracker - YOLOv8", blank_frame)
                else:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, f"Esperando imagen... Objetivo: {node.target_class if node.target_class else 'None'}", 
                               (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Object Tracker - YOLOv8", blank_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()