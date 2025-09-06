#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import random
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import tkinter as tk
from tkinter import simpledialog
import subprocess


class ObjectTracker:
    """Clase para mejorar el seguimiento de objetos detectados"""
    
    def __init__(self, max_distance=2.0, max_time_missing=3.0):
        self.objects = {}  # {track_id: object_info}
        self.next_id = 0
        self.max_distance = max_distance  # Distancia máxima para considerar el mismo objeto
        self.max_time_missing = max_time_missing  # Tiempo máximo sin detectar antes de eliminar
        
    def update(self, detections, current_time):
        """
        Actualiza el tracker con nuevas detecciones sin predicción de movimiento.
        Las posiciones no se actualizan si el objeto no es visto.
        """
        # Marcar todos los objetos como no vistos en este frame
        for obj in self.objects.values():
            obj['seen_this_frame'] = False

        # Para cada detección, buscar coincidencia con objeto existente
        for detection in detections:
            label = detection['label']
            pos = detection['pos']
            confidence = detection['confidence']

            best_match_id = None
            best_distance = float('inf')

            for track_id, obj in self.objects.items():
                if obj['label'] == label:
                    distance = math.hypot(pos[0] - obj['pos'][0], pos[1] - obj['pos'][1])

                    if distance < self.max_distance and distance < best_distance:
                        best_distance = distance
                        best_match_id = track_id

            if best_match_id is not None:
                # Actualizar objeto existente
                obj = self.objects[best_match_id]

                # Suavizado opcional de posición (puedes quitar si no lo deseas)
                alpha_pos = 0.4
                obj['pos'] = (
                    (1 - alpha_pos) * obj['pos'][0] + alpha_pos * pos[0],
                    (1 - alpha_pos) * obj['pos'][1] + alpha_pos * pos[1]
                )

                obj['confidence'] = max(obj['confidence'], confidence)
                obj['last_update'] = current_time
                obj['last_seen'] = current_time
                obj['seen_this_frame'] = True
                obj['detection_count'] += 1
                obj['active'] = True  # Reactivar si estaba inactivo

            else:
                # Crear nuevo objeto
                self.objects[self.next_id] = {
                    'id': self.next_id,
                    'label': label,
                    'pos': pos,
                    'velocity': (0.0, 0.0),
                    'confidence': confidence,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'last_update': current_time,
                    'active': True,
                    'seen_this_frame': True,
                    'detection_count': 1
                }
                self.next_id += 1

        # Desactivar objetos que no fueron vistos recientemente
        for obj in self.objects.values():
            if not obj['seen_this_frame']:
                if current_time - obj['last_seen'] > self.max_time_missing:
                    obj['active'] = False
                # NO se actualiza la posición si no fue visto

    
    def _predict_position(self, obj, current_time):
        """Predice la posición del objeto basada en su velocidad"""
        dt = current_time - obj['last_update']
        predicted_x = obj['pos'][0] + obj['velocity'][0] * dt
        predicted_y = obj['pos'][1] + obj['velocity'][1] * dt
        return (predicted_x, predicted_y)
    
    def get_active_objects(self):
        """Retorna solo los objetos activos"""
        return {tid: obj for tid, obj in self.objects.items() if obj['active']}
    
    def get_all_objects(self):
        """Retorna todos los objetos (activos e inactivos)"""
        return self.objects.copy()

class MapObjectDetector(Node):
    def __init__(self):
        super().__init__('map_object_detector')

        self.bridge = CvBridge()
        self.map_received = False
        self.map_image = None
        self.map_resolution = None
        self.map_origin = None

        # Tracker mejorado
        self.tracker = ObjectTracker(max_distance=1.5, max_time_missing=3.0)
        self.class_colors = {}

        # YOLO model
        self.model = YOLO('yolov8n.pt')

        # Camera parameters
        self.camera_intrinsics = None
        self.camera_fov_rad = math.radians(90)

        # Setup TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_frame = 'front_camera' 
        self.lidar_frame = 'base_link'
        self.map_frame = 'map'

        # Variables para selección de clase
        self.target_class = None
        self.class_selected = False
        self.windows_created = False

        # Variables para guardado de datos
        self.session_start_time = datetime.now()
        self.session_data = {
            'session_info': {
                'start_time': self.session_start_time.isoformat(),
                'target_class': None,
                'map_resolution': None,
                'map_origin': None
            },
            'detections': [],
            'final_objects': {}
        }

        # Queue sizes and QoS
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        # Setup message filter for synchronized processing
        self.image_sub = Subscriber(self, Image, '/camera/image_raw', qos_profile=sensor_qos)
        self.scan_sub = Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
        
        # Synchronize messages with timestamps within 0.1 seconds
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Solicitar clase al usuario al inicio
        self.request_target_class()
        
        self.get_logger().info("Detector iniciado. Esperando selección de clase...")

    def request_target_class(self):
        """Solicita al usuario que ingrese la clase objetivo"""
        try:
            root = tk.Tk()
            root.withdraw()

            available_classes = list(self.model.names.values())
            class_lookup = {cls.lower(): cls for cls in available_classes}
            example_classes = sorted(available_classes)[:5]
            classes_text = ", ".join([f"{cls}" for cls in example_classes])
            
            prompt = f"Ingrese la clase a detectar (o 'All' para todas):\n\nEjemplos de clases:\n{classes_text}..."

            user_input = simpledialog.askstring(
                "Selección de Clase", 
                prompt,
                initialvalue="person"
            )

            root.destroy()

            if user_input:
                user_input_lower = user_input.lower()

                if user_input_lower == 'all':
                    self.target_class = 'all'
                    self.get_logger().info("Detectando TODAS las clases")
                elif user_input_lower in class_lookup:
                    self.target_class = class_lookup[user_input_lower]
                    self.get_logger().info(f"Detectando clase: {self.target_class}")
                else:
                    self.get_logger().warn(f"Clase '{user_input}' no encontrada. Detectando todas las clases.")
                    self.target_class = 'all'

                # Actualizar datos de sesión
                self.session_data['session_info']['target_class'] = self.target_class
                self.class_selected = True
            else:
                self.get_logger().info("No se seleccionó clase. Cerrando...")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error al solicitar clase: {e}")
            self.target_class = 'all'
            self.class_selected = True

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_intrinsics = np.array(msg.k).reshape(3, 3)
        fx = self.camera_intrinsics[0, 0]
        image_width = msg.width
        self.camera_fov_rad = 2 * math.atan2(image_width / 2, fx)
        self.get_logger().info(f"Camera info received. FOV: {math.degrees(self.camera_fov_rad):.2f} degrees")

    def map_callback(self, msg):
        """Process the map data."""
        width, height = msg.info.width, msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position

        # Actualizar datos de sesión
        self.session_data['session_info']['map_resolution'] = self.map_resolution
        self.session_data['session_info']['map_origin'] = {
            'x': self.map_origin.x,
            'y': self.map_origin.y,
            'z': self.map_origin.z
        }

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        map_img = np.zeros((height, width, 3), dtype=np.uint8)
        map_img[data == 0] = [255, 255, 255]
        map_img[data == 100] = [0, 0, 0]
        map_img[data == -1] = [150, 150, 150]

        self.map_image = cv2.flip(map_img, 0)
        self.map_received = True
        self.get_logger().info(f"Mapa recibido: {width}x{height}, resol: {self.map_resolution}")

    def should_process_detection(self, label):
        """Determina si se debe procesar una detección basada en la clase objetivo"""
        if self.target_class == 'all':
            return True
        return label == self.target_class

    def get_color_for_class(self, class_name):
        """Generate and return a consistent color for each class."""
        if class_name not in self.class_colors:
            self.class_colors[class_name] = (
                random.randint(0, 200), 
                random.randint(0, 200), 
                random.randint(0, 200)
            )
        return self.class_colors[class_name]

    def image_point_to_ray(self, u, v, frame_w, frame_h):
        """Convert image point to a 3D ray direction and corresponding LiDAR angle."""
        if self.camera_intrinsics is not None:
            fx = self.camera_intrinsics[0, 0]
            fy = self.camera_intrinsics[1, 1]
            cx = self.camera_intrinsics[0, 2]
            cy = self.camera_intrinsics[1, 2]
            
            x = (u - cx) / fx
            y = (v - cy) / fy
            
            ray = np.array([1.0, x, y])
            ray = ray / np.linalg.norm(ray)
            
            camera_angle = math.atan2(x, 1.0)
            lidar_angle = -camera_angle
            
            if lidar_angle > math.pi:
                lidar_angle -= 2 * math.pi
            elif lidar_angle < -math.pi:
                lidar_angle += 2 * math.pi
                
            return ray, lidar_angle
        else:
            rel_x = (u - frame_w / 2) / (frame_w / 2)
            angle = rel_x * (self.camera_fov_rad / 2)
            angle = -angle
            ray = np.array([math.cos(angle), math.sin(angle), 0.0])
            return ray, angle

    def get_closest_valid_range(self, scan, target_index, window_size=3):
        """Get closest valid range measurement around target index."""
        if 0 <= target_index < len(scan.ranges):
            if np.isfinite(scan.ranges[target_index]) and scan.ranges[target_index] > 0.1:
                return scan.ranges[target_index]
                
        ranges = []
        for i in range(max(0, target_index - window_size), min(len(scan.ranges), target_index + window_size + 1)):
            if np.isfinite(scan.ranges[i]) and scan.ranges[i] > 0.1:
                ranges.append((abs(i - target_index), scan.ranges[i]))
                
        if ranges:
            ranges.sort()
            return ranges[0][1]
            
        return None
    
    def draw_robot_position(self, display_map, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.lidar_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y

            mx = int((x - self.map_origin.x) / self.map_resolution)
            my = self.map_image.shape[0] - int((y - self.map_origin.y) / self.map_resolution)

            if 0 <= mx < display_map.shape[1] and 0 <= my < display_map.shape[0]:
                color = (255, 0, 255)
                size = 1
                cv2.line(display_map, (mx - size, my), (mx + size, my), color, 1)
                cv2.line(display_map, (mx, my - size), (mx, my + size), color, 1)
        except TransformException as ex:
            self.get_logger().warn(f"No se pudo obtener la posición del robot: {ex}")

    def create_windows_if_needed(self):
        """Crea las ventanas de OpenCV solo cuando sea necesario"""
        if not self.windows_created:
            cv2.namedWindow("Detecciones en mapa", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Detecciones YOLO", cv2.WINDOW_NORMAL)
            self.windows_created = True
            self.get_logger().info("Ventanas creadas. Presiona 'q' para cerrar, 's' para guardar.")

    def save_detection_data(self, detections, timestamp):
        """Guarda los datos de detección en el historial"""
        frame_data = {
            'timestamp': timestamp,
            'detections_count': len(detections),
            'detections': detections
        }
        self.session_data['detections'].append(frame_data)


    def save_final_data(self):
        """Guarda los datos finales en JSON, PNG y ejecuta map_saver_cli"""
        try:
            # Actualizar objetos finales
            all_objects = self.tracker.get_all_objects()
            self.session_data['final_objects'] = {}
            
            for track_id, obj in all_objects.items():
                self.session_data['final_objects'][str(track_id)] = {
                    'id': obj['id'],
                    'label': obj['label'],
                    'final_position': {
                        'x': obj['pos'][0],
                        'y': obj['pos'][1]
                    },
                    'map_coordinates': {
                        'mx': int((obj['pos'][0] - self.map_origin.x) / self.map_resolution) if self.map_origin else None,
                        'my': self.map_image.shape[0] - int((obj['pos'][1] - self.map_origin.y) / self.map_resolution) if self.map_origin and self.map_image is not None else None
                    },
                    'confidence': obj['confidence'],
                    'first_seen': obj['first_seen'],
                    'last_seen': obj['last_seen'],
                    'total_detections': obj['detection_count'],
                    'active': obj['active'],
                    'session_duration': obj['last_seen'] - obj['first_seen']
                }

            # Actualizar información de sesión
            self.session_data['session_info']['end_time'] = datetime.now().isoformat()
            self.session_data['session_info']['total_frames'] = len(self.session_data['detections'])
            
            # Contar objetos por clase
            class_summary = defaultdict(int)
            for obj in all_objects.values():
                if self.should_process_detection(obj['label']):
                    class_summary[obj['label']] += 1
            
            self.session_data['session_info']['summary'] = dict(class_summary)

            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"Y_detections_{timestamp}.json"
            png_filename = f"Y_mapa_detecciones_{timestamp}.png"
            
            # Guardar archivo JSON
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            
            # Crear mapa con detecciones y guardarlo como PNG
            display_map = self.map_image.copy()
            
            # Dibujar objetos en el mapa
            for track_id, obj in all_objects.items():
                if not self.should_process_detection(obj['label']):
                    continue
                    
                x, y = obj['pos']
                mx = int((x - self.map_origin.x) / self.map_resolution)
                my = self.map_image.shape[0] - int((y - self.map_origin.y) / self.map_resolution)

                if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                    color = self.get_color_for_class(obj['label'])
                    
                    if obj['active']:
                        cv2.circle(display_map, (mx, my), 1, color, -1)
                        text_color = color
                    else:
                        cv2.circle(display_map, (mx, my), 1, color, -1)
                        text_color = color
                    
                    cv2.putText(display_map, f"{obj['label']}_{track_id}", (mx + 5, my - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, text_color, 1)
            
            # Guardar mapa con detecciones como PNG
            resized_map = cv2.resize(display_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(png_filename, resized_map)
            
            # Ejecutar map_saver_cli
            try:
                map_name = f'Y_saved_map_{timestamp}'

                # Ejecutar el comando
                result = subprocess.run(
                    ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_name],
                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.get_logger().info("map_saver_cli ejecutado exitosamente")
                else:
                    self.get_logger().warn(f"map_saver_cli falló: {result.stderr}")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("map_saver_cli timeout")
            except Exception as e:
                self.get_logger().error(f"Error ejecutando map_saver_cli: {e}")
            
            self.get_logger().info(f"Guardado completado:")
            self.get_logger().info(f"  - JSON: {json_filename}")
            self.get_logger().info(f"  - PNG: {png_filename}")
            self.get_logger().info(f"  - map_saver_cli ejecutado")
            
            # Estadísticas resumidas
            total_objects = len(all_objects)
            active_objects = len(self.tracker.get_active_objects())
            self.get_logger().info(f"Total objetos: {total_objects}, Activos: {active_objects}")
            
            return json_filename
            
        except Exception as e:
            self.get_logger().error(f"Error al guardar datos: {e}")
            return None


    def synchronized_callback(self, img_msg, scan_msg):
        """Process synchronized image and laser scan data."""
        if not self.class_selected:
            return
            
        if not self.map_received:
            return

        self.create_windows_if_needed()

        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            frame_h, frame_w = frame.shape[:2]
            frame_viz = frame.copy()

            angle_min = scan_msg.angle_min
            angle_max = scan_msg.angle_max
            angle_inc = scan_msg.angle_increment

            results = self.model(frame)[0]
            current_time = time.time()
            display_map = self.map_image.copy()

            self.draw_robot_position(display_map, img_msg.header.stamp)

            # Procesar detecciones para el tracker
            detections_for_tracker = []
            frame_detections = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if not self.should_process_detection(label):
                    continue

                color = self.get_color_for_class(label)
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_viz, f"{label} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if conf < 0.6:
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame_viz, (center_x, center_y), 4, (0, 255, 0), -1)

                _, angle_rad = self.image_point_to_ray(center_x, center_y, frame_w, frame_h)
                
                if angle_min <= angle_rad <= angle_max:
                    index = int(round((angle_rad - angle_min) / angle_inc))
                    index = max(0, min(index, len(scan_msg.ranges) - 1))
                    
                    distance = self.get_closest_valid_range(scan_msg, index, window_size=5)
                    if distance is None:
                        continue

                    point_lidar = PointStamped()
                    point_lidar.header.stamp = img_msg.header.stamp
                    point_lidar.header.frame_id = self.lidar_frame
                    
                    point_lidar.point.x = distance * math.cos(angle_rad)
                    point_lidar.point.y = distance * math.sin(angle_rad)
                    point_lidar.point.z = 0.0

                    try:
                        transform = self.tf_buffer.lookup_transform(
                            self.map_frame, 
                            self.lidar_frame, 
                            rclpy.time.Time(), 
                            timeout=rclpy.duration.Duration(seconds=0.3)
                        )
                        point_map = do_transform_point(point_lidar, transform)
                        x, y = point_map.point.x, point_map.point.y

                        mx = int((x - self.map_origin.x) / self.map_resolution)
                        my = self.map_image.shape[0] - int((y - self.map_origin.y) / self.map_resolution)

                        if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                            # Agregar detección para el tracker
                            detections_for_tracker.append({
                                'label': label,
                                'pos': (x, y),
                                'confidence': conf
                            })
                            
                            # Agregar detección para el historial
                            frame_detections.append({
                                'label': label,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'world_position': {'x': x, 'y': y},
                                'map_coordinates': {'mx': mx, 'my': my},
                                'distance': distance,
                                'angle_deg': math.degrees(angle_rad)
                            })

                    except TransformException as ex:
                        self.get_logger().warn(f"Transform error: {ex}")

            # Actualizar tracker
            self.tracker.update(detections_for_tracker, current_time)
            
            # Guardar datos de este frame
            self.save_detection_data(frame_detections, current_time)

            # Visualizar objetos tracked
            active_objects = self.tracker.get_active_objects()
            all_objects = self.tracker.get_all_objects()
            
            for track_id, obj in all_objects.items():
                if not self.should_process_detection(obj['label']):
                    continue
                    
                x, y = obj['pos']
                mx = int((x - self.map_origin.x) / self.map_resolution)
                my = self.map_image.shape[0] - int((y - self.map_origin.y) / self.map_resolution)

                if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                    color = self.get_color_for_class(obj['label'])
                    
                    # Diferentes estilos para activos/inactivos
                    if obj['active']:
                        cv2.circle(display_map, (mx, my), 1, color, -1)
                        text_color = color
                    else:
                        cv2.circle(display_map, (mx, my), 1, (128, 128, 128), -1)
                        text_color = (128, 128, 128)
                    
                    # Mostrar ID y etiqueta
                    cv2.putText(display_map, f"{obj['label']}_{track_id}", (mx + 5, my - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, text_color, 1)


            # Display results
            resized_map = cv2.resize(display_map, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            resized_frame = cv2.resize(frame_viz, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            
            # Crosshair
            h, w = resized_frame.shape[:2]
            cv2.line(resized_frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            cv2.line(resized_frame, (0, h//2), (w, h//2), (0, 255, 0), 1)

            # Leyenda mejorada
            start_x = resized_map.shape[1] - 150
            start_y = 20

            # Información de la sesión
            target_text = f"Target: {self.target_class}"
            cv2.putText(resized_map, target_text, (start_x, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            
            cv2.imshow("Detecciones en mapa", resized_map)
            cv2.imshow("Detecciones YOLO", resized_frame)
            cv2.resizeWindow("Detecciones en mapa", 900, 600)
            cv2.resizeWindow("Detecciones YOLO", 900, 600)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Guardar datos finales antes de cerrar
                self.save_final_data()
                rclpy.shutdown()
            elif key == ord('s'):
                # Guardar mapa e imágenes
                timestamp = int(time.time())
                png_filename = f"mapa_{timestamp}.png"
                json_filename = self.save_final_data()

                # Guardar como PNG
                cv2.imwrite(png_filename, resized_map)
                self.get_logger().info(f"Mapa guardado como PNG: {png_filename}")
                if json_filename:
                    self.get_logger().info(f"Datos guardados en: {json_filename}")

            total_detections = len(results.boxes)
            filtered_detections = len(detections_for_tracker)
            tracked_objects = len(active_objects)
            
            self.get_logger().info(f"Frame: {total_detections} detectados, {filtered_detections} válidos, {tracked_objects} objetos activos")
            
        except Exception as e:
            self.get_logger().error(f"Error en procesamiento: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = MapObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Guardar datos finales al cerrar
        if node.class_selected:
            node.save_final_data()
        node.destroy_node()
        cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()