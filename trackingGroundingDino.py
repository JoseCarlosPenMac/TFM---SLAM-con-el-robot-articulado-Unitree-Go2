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
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import simpledialog, messagebox
import time

from groundingdino.util.inference import load_model, predict

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        
        self.bridge = CvBridge()
        
        # Modelo GroundingDINO
        CONFIG_PATH = "/home/josecarlos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        WEIGHTS_PATH = "/home/josecarlos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(self.device)
            print("✅ Modelo GroundingDINO cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return
            
        # Parámetros de detección
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        self.transform = T.Compose([T.ToTensor()])
        
        # Clase objetivo
        self.target_class = None
        self.class_selected = False
        self.request_target_class()
        
        # Control de movimiento
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Variables de estado
        self.object_detected = False
        self.object_center_x = None
        self.last_object_center_x = None  # Para recordar última posición
        self.image_center_x = None
        
        # Variables de distancia LIDAR
        self.front_distance = float('inf')          # Para tracking (mínimo)
        self.front_value_percentil = float('inf')   # Para navegación autónoma (percentil)
        self.left_value = float('inf')
        self.right_value = float('inf')
        
        # Parámetros de tracking
        self.CENTERING_THRESHOLD = 250  # pixels de tolerancia para considerar centrado
        self.MIN_DISTANCE = 1.5       # Distancia mínima al objeto (metros)
        
        # Estados de búsqueda
        self.search_start_time = None
        self.SEARCH_DURATION = 20.0   # 20 segundos de búsqueda girando
        self.search_direction = 0.15
        self.is_searching = False
        self.is_autonomous_nav = False

        # Variables para navegación autónoma (consistentes con obstacle_avoider)
        self.is_reversing = False
        self.reverse_timer = 0
        self.turning_direction = None  # Añadido para consistencia
        
        # Visualización matplotlib
        self.fig = None
        self.ax = None
        self.matplotlib_initialized = False
        
        # Suscripciones
        lidar_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile=lidar_qos)
        
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, sensor_qos)
        
        # Timer de control
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Object Tracker iniciado. Esperando selección de clase...")

    def request_target_class(self):
        """Solicitar clase objetivo mediante ventana tkinter"""
        try:
            # Crear ventana tkinter
            root = tk.Tk()
            root.withdraw()  # Ocultar ventana principal
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
            
            user_input = simpledialog.askstring(
                "Object Tracker - GroundingDINO",
                "Ingrese la clase a detectar (ej: 'person', 'car', 'bottle'):",
                parent=root
            )
            
            if user_input and user_input.strip():
                self.target_class = user_input.strip()
                messagebox.showinfo("Configuración", f"✅ Detectando y siguiendo: {self.target_class}")
            else:
                self.target_class = 'person'
                messagebox.showinfo("Configuración", "Usando clase por defecto: person")
            
            self.class_selected = True
            root.destroy()
            print(f"✅ Detectando: {self.target_class}")
            
        except Exception as e:
            self.get_logger().error(f"Error con ventana tkinter: {e}")
            # Fallback a terminal si tkinter falla
            print("\n" + "="*50)
            print("OBJECT TRACKER - GROUNDING DINO")
            print("="*50)
            print("No se pudo abrir ventana gráfica. Usando entrada por terminal:")
            target = input("Ingrese la clase a detectar (ej: 'person', 'car') [default: person]: ").strip()
            
            self.target_class = target if target else 'person'
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
        """Detectar objetos usando GroundingDINO"""
        try:
            frame_h, frame_w = frame.shape[:2]
            small_frame = cv2.resize(frame, (640, 480))
            image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor[0],
                    caption=self.target_class,
                    box_threshold=self.BOX_THRESHOLD,
                    text_threshold=self.TEXT_THRESHOLD
                )

            detections = []
            for box, phrase, logit in zip(boxes, phrases, logits):
                confidence = float(logit)
                if confidence < 0.6:
                    continue
                    
                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * frame_w)
                y1 = int((cy - bh / 2) * frame_h)
                x2 = int((cx + bw / 2) * frame_w)
                y2 = int((cy + bh / 2) * frame_h)
                
                center_x = (x1 + x2) // 2
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center_x': center_x,
                    'confidence': confidence,
                    'label': phrase
                })
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"Error en detección: {e}")
            return []

    def init_matplotlib(self):
        """Inicializar matplotlib para visualización"""
        if not self.matplotlib_initialized:
            plt.close('all')
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.set_title("Object Tracking View")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            plt.ion()
            plt.show(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.matplotlib_initialized = True

    def get_color_for_class(self, class_name):
        """Generar color único para cada clase"""
        import hashlib
        hex_dig = hashlib.md5(class_name.encode()).hexdigest()
        r = int(hex_dig[0:2], 16) / 255.0
        g = int(hex_dig[2:4], 16) / 255.0
        b = int(hex_dig[4:6], 16) / 255.0
        if r + g + b < 1.2:
            r, g, b = min(1.0, r + 0.4), min(1.0, g + 0.4), min(1.0, b + 0.4)
        return (r, g, b)

    def image_callback(self, msg):
        """Procesar imagen y detectar objetos"""
        # No procesar hasta que se seleccione la clase
        if not self.class_selected:
            return
            
        self.init_matplotlib()
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_h, frame_w = frame.shape[:2]
            self.image_center_x = frame_w // 2
            
            detections = self.detect_objects(frame)
            
            if len(detections) > 0:
                # Punto 3: Solo seguir al objeto con mejor porcentaje
                best_detection = max(detections, key=lambda x: x['confidence'])
                self.object_detected = True
                self.last_object_center_x = self.object_center_x  # Guardar posición anterior
                self.object_center_x = best_detection['center_x']
                
                # Resetear estados de búsqueda
                self.is_searching = False
                self.is_autonomous_nav = False
                self.search_start_time = None
                
                print(f"✅ Objeto detectado: {best_detection['label']} (conf: {best_detection['confidence']:.2f}) en x={self.object_center_x}")
            else:
                if self.object_detected:
                    # Punto 4: Objeto desapareció, determinar dirección de búsqueda
                    if self.object_center_x is not None and self.image_center_x is not None:
                        if self.object_center_x > self.image_center_x:
                            print("🔍 Objeto salió por la derecha - buscando a la derecha")
                            self.search_direction = -0.3  # Girar a la derecha
                        else:
                            print("🔍 Objeto salió por la izquierda - buscando a la izquierda")
                            self.search_direction = 0.3   # Girar a la izquierda
                    else:
                        self.search_direction = 0.3  # Dirección por defecto
                
                self.object_detected = False
                self.object_center_x = None
                
                # Iniciar búsqueda si no estaba ya buscando
                if not self.is_searching and not self.is_autonomous_nav:
                    self.is_searching = True
                    self.search_start_time = time.time()
                    print("🔄 Iniciando búsqueda giratoria por 20 segundos")
            
            # VISUALIZACIÓN CON MATPLOTLIB
            self.ax.clear()
            camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.ax.imshow(camera_rgb)
            
            # Título dinámico según el estado
            if self.object_detected and self.front_distance <= self.MIN_DISTANCE:
                title = f"🛑 PARADO - Objeto alcanzado: {self.target_class} (Dist: {self.front_distance:.1f}m)"
            elif self.object_detected:
                title = f"🎯 TRACKING: {self.target_class} (Dist: {self.front_distance:.1f}m)"
            elif self.is_searching:
                elapsed = time.time() - self.search_start_time if self.search_start_time else 0
                title = f"🔄 BUSCANDO: {self.target_class} ({elapsed:.1f}/{self.SEARCH_DURATION}s)"
            elif self.is_autonomous_nav:
                title = f"🗺️ NAVEGACIÓN AUTÓNOMA - Buscando: {self.target_class}"
            else:
                title = f"🔍 BÚSQUEDA INICIAL - Objetivo: {self.target_class}"
            
            self.ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Dibujar detecciones
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                color = self.get_color_for_class(det['label'])
                
                # Caja de detección
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=3, edgecolor=color, facecolor='none')
                self.ax.add_patch(rect)
                
                # Etiqueta
                self.ax.text(x1, y1 - 10, f"{det['label']} {det['confidence']:.2f}",
                           fontsize=10, color=color, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Centro del objeto
                center_x, center_y = det['center_x'], (y1 + y2) // 2
                self.ax.plot(center_x, center_y, 'ro', markersize=8)
            
            # Líneas de referencia para centrado
            self.ax.axvline(x=self.image_center_x, color='green', linewidth=2, alpha=0.7, label='Centro')
            
            # Zona de tolerancia para centrado
            left_bound = self.image_center_x - self.CENTERING_THRESHOLD
            right_bound = self.image_center_x + self.CENTERING_THRESHOLD
            self.ax.axvline(x=left_bound, color='yellow', linewidth=1, linestyle='--', alpha=0.7)
            self.ax.axvline(x=right_bound, color='yellow', linewidth=1, linestyle='--', alpha=0.7)
            
            # Información de estado
            if self.object_detected:
                error = abs(self.object_center_x - self.image_center_x)
                centered = error <= self.CENTERING_THRESHOLD
                status_text = f"🎯 Centrado: {centered} | Error: {error}px | Detecciones: {len(detections)}"
                status_color = 'green'
            elif self.is_searching:
                elapsed = time.time() - self.search_start_time if self.search_start_time else 0
                status_text = f"🔄 Búsqueda giratoria: {elapsed:.1f}s restantes"
                status_color = 'orange'
            elif self.is_autonomous_nav:
                status_text = f"🗺️ Navegación autónoma | Distancia frontal: {self.front_value_percentil:.1f}m"
                status_color = 'blue'
            else:
                status_text = f"🔍 Búsqueda inicial | Detecciones: {len(detections)}"
                status_color = 'purple'
            
            self.ax.text(10, 30, status_text, fontsize=9, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8))
            
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
                
        except Exception as e:
            self.get_logger().error(f"Error procesando imagen: {e}")

    def control_loop(self):
        """Loop principal de control del robot"""
        # No mover el robot hasta que se seleccione la clase
        if not self.class_selected:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return
            
        cmd = Twist()

        # Punto 1: Si encuentra objeto, moverse hacia él
        if self.object_detected and self.object_center_x is not None:
            print(f"🎯 TRACKING - Distancia frontal: {self.front_distance:.2f}m")
            
            # Calcular error de centrado
            error_x = self.object_center_x - self.image_center_x
            
            # Verificar si está demasiado cerca
            if self.front_distance <= self.MIN_DISTANCE:
                # Parado por distancia, pero puede girar para mantener centrado
                cmd.linear.x = 0.0
                
                if abs(error_x) > self.CENTERING_THRESHOLD:
                    # Objeto no centrado - girar para centrarlo (sin avanzar)
                    if error_x > 0:
                        cmd.angular.z = -0.2  # Girar a la derecha
                    else:
                        cmd.angular.z = 0.2   # Girar a la izquierda
                    print(f"🛑 PARADO pero centrando objeto - error: {error_x}")
                else:
                    cmd.angular.z = 0.0
                    print(f"🛑 PARADO - Distancia objetivo alcanzada y objeto centrado: {self.front_distance:.2f}m")
            else:
                # Distancia segura - puede avanzar y centrar
                if abs(error_x) > self.CENTERING_THRESHOLD:
                    # Objeto no centrado - girar para centrarlo
                    if error_x > 0:
                        cmd.angular.z = -0.2  # Girar a la derecha
                    else:
                        cmd.angular.z = 0.2   # Girar a la izquierda
                    
                    cmd.linear.x = 0.1  # Avanzar lentamente mientras centra
                    print(f"🎯 Centrando objeto - error: {error_x}")
                else:
                    # Objeto centrado - avanzar hacia él
                    cmd.linear.x = 0.3
                    cmd.angular.z = -0.05
                    print("✅ Objeto centrado - avanzando")
        
        # Punto 2: Búsqueda giratoria por 20 segundos
        elif self.is_searching:
            current_time = time.time()
            elapsed_time = current_time - self.search_start_time
            
            if elapsed_time < self.SEARCH_DURATION:
                cmd.linear.x = 0.0
                cmd.angular.z = self.search_direction
                print(f"🔄 Buscando girando - {elapsed_time:.1f}/{self.SEARCH_DURATION}s")
            else:
                # Terminar búsqueda, iniciar navegación autónoma
                self.is_searching = False
                self.is_autonomous_nav = True
                print("🗺️ Búsqueda terminada - iniciando navegación autónoma")
        
        # NAVEGACIÓN AUTÓNOMA CORREGIDA
        elif self.is_autonomous_nav:
            print(f"NAVEGACIÓN AUTONOMA: Front: {self.front_value_percentil:.2f}, Left: {self.left_value:.2f}, Right: {self.right_value:.2f}")

            # Si está en modo de retroceso
            if self.is_reversing:
                cmd.linear.x = -0.2
                self.reverse_timer += 1

                # Después de un tiempo (aproximadamente 2 segundos a 10Hz)
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
                # Si hay obstáculo a la izquierda, girar a la derecha
                elif left_blocked:
                    self.turning_direction = -0.2
                    cmd.angular.z = self.turning_direction
                # Si hay obstáculo a la derecha, girar a la izquierda
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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parar el robot
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()