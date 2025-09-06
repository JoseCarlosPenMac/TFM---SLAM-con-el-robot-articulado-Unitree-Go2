#!/usr/bin/env python3
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

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
import numpy as np
import math
import random
import time
import json
import os
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tkinter as tk
from tkinter import simpledialog, messagebox
import subprocess

import cv2
import torch
import torchvision.transforms as T
from groundingdino.util.inference import load_model, predict

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

        # GroundingDINO model setup
        CONFIG_PATH = "/home/josecarlos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        WEIGHTS_PATH = "/home/josecarlos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Usando dispositivo: {self.device}")
        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(self.device)
        
        # GroundingDINO parameters
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        self.transform = T.Compose([T.ToTensor()])
        self.skip_frames = 15
        self.min_detection_interval = 3.0
        self.last_detection_time = 0
        self.frame_counter = 0

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
        
        # Matplotlib setup - SOLO UNA INICIALIZACIÓN
        self.fig = None
        self.ax_map = None
        self.ax_camera = None
        self.matplotlib_initialized = False

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
            queue_size=5, 
            slop=0.5
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.last_frame_detections = []


        # Solicitar clase al usuario al inicio
        self.request_target_class()
        
        self.get_logger().info("Detector iniciado. Esperando selección de clase...")
    
    def init_matplotlib(self):
        """Inicializar matplotlib UNA SOLA VEZ"""
        if not self.matplotlib_initialized:
            # Cerrar cualquier figura existente
            plt.close('all')
            
            # Configurar matplotlib para optimización
            plt.ioff()  # Desactivar modo interactivo inicialmente
            
            # Crear UNA SOLA figura
            self.fig, (self.ax_map, self.ax_camera) = plt.subplots(1, 2, figsize=(15, 8))
            self.fig.suptitle("Map Object Detector - GroundingDINO")
            
            # Configuración inicial de los ejes
            self.ax_map.set_title("Map View")
            self.ax_camera.set_title("Camera View")
            
            # Optimizaciones para fluidez
            self.ax_map.set_xticks([])
            self.ax_map.set_yticks([])
            self.ax_camera.set_xticks([])
            self.ax_camera.set_yticks([])
            
            # Activar modo interactivo DESPUÉS de crear todo
            plt.ion()
            plt.show(block=False)
            
            # Forzar dibujo inicial
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.matplotlib_initialized = True
            self.get_logger().info("Matplotlib inicializado. Cierra la ventana para terminar.")

            
    def request_target_class(self):
        """Solicita al usuario que ingrese la clase objetivo usando Tkinter"""
        try:
            # Crear ventana raíz oculta
            root = tk.Tk()
            root.withdraw()  # Ocultar ventana principal
            
            # Solicitar input al usuario
            user_input = simpledialog.askstring(
                "Detector de Objetos",
                "Ingrese la clase a detectar (ej: 'person', 'bottle', 'car', etc.):",
                parent=root
            )
            
            if user_input and user_input.strip():
                self.target_class = user_input.strip()
                self.get_logger().info(f"Detectando clase: {self.target_class}")
                self.session_data['session_info']['target_class'] = self.target_class
                self.class_selected = True
                messagebox.showinfo("Configuración", f"Detectando clase: {self.target_class}")
            else:
                self.target_class = 'person'
                self.class_selected = True
                messagebox.showinfo("Configuración", "Usando clase por defecto: person")
            
            root.destroy()
            
        except Exception as e:
            self.get_logger().error(f"Error al solicitar clase: {e}")
            self.target_class = 'person'
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

        self.map_image = map_img
        self.map_received = True
        self.get_logger().info(f"Mapa recibido: {width}x{height}, resol: {self.map_resolution}")

    def get_color_for_class(self, class_name):
        """Generate and return a consistent color for each class."""
        if class_name not in self.class_colors:
            self.class_colors[class_name] = (
                random.randint(50, 255), 
                random.randint(50, 255), 
                random.randint(50, 255)
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

    
    def save_detection_data(self, detections, timestamp):
        """Guarda los datos de detección en el historial"""
        frame_data = {
            'timestamp': timestamp,
            'detections_count': len(detections),
            'detections': detections
        }
        self.session_data['detections'].append(frame_data)

    def save_final_data(self):
        """Guarda solo el mapa PNG, JSON y ejecuta map_saver_cli"""
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
                        'my': int((obj['pos'][1] - self.map_origin.y) / self.map_resolution) if self.map_origin else None
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
                class_summary[obj['label']] += 1
            
            self.session_data['session_info']['summary'] = dict(class_summary)

            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Guardar archivo JSON
            json_filename = f"GD_detections_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            self.get_logger().info(f"JSON guardado: {json_filename}")
            
            # 2. Guardar SOLO el mapa con detecciones (ax_map) en PNG
            if self.matplotlib_initialized and self.fig is not None:
                map_filename = f"GD_map_detections_{timestamp}.png"
                
                # Guardar solo el subplot del mapa
                extent = self.ax_map.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(map_filename, bbox_inches=extent, dpi=150, 
                            facecolor='white', edgecolor='none')
                self.get_logger().info(f"Mapa con detecciones guardado: {map_filename}")
            
            # 3. Ejecutar comando map_saver_cli
            try:
                map_name = f"GD_saved_map_{timestamp}"
                cmd = ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", map_name]
                
                self.get_logger().info(f"Ejecutando: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.get_logger().info(f"Mapa guardado exitosamente como: {map_name}")
                else:
                    self.get_logger().error(f"Error al guardar mapa: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.get_logger().error("Timeout al ejecutar map_saver_cli")
            except Exception as e:
                self.get_logger().error(f"Error al ejecutar map_saver_cli: {e}")
            
            # Estadísticas resumidas
            total_objects = len(all_objects)
            active_objects = len(self.tracker.get_active_objects())
            self.get_logger().info(f"Guardado completado - Total objetos: {total_objects}, Activos: {active_objects}")
            
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

        # Inicializar matplotlib UNA SOLA VEZ
        self.init_matplotlib()

        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            frame_h, frame_w = frame.shape[:2]
            
            # Check if matplotlib window is closed - VERIFICACIÓN OPTIMIZADA
            if self.matplotlib_initialized and (self.fig is None or not plt.fignum_exists(self.fig.number)):
                self.save_final_data()
                rclpy.shutdown()
                return

            current_time = time.time()
            self.frame_counter += 1

            # Determinar si ejecutar detección
            run_detection = (
                self.frame_counter % self.skip_frames == 0 or
                (current_time - self.last_detection_time) > self.min_detection_interval
            )

            detections_for_tracker = []
            frame_detections = []

            if run_detection:
                # Definir tamaño reducido para GroundingDINO (ajusta según tu hardware)
                small_width = 320
                small_height = 240
                
                # Calcular factores de escala
                scale_x = frame_w / small_width
                scale_y = frame_h / small_height
                
                # Redimensionar SOLO para el modelo
                small_frame = cv2.resize(frame, (small_width, small_height))
                image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

                # Ejecutar GroundingDINO con imagen pequeña
                with torch.no_grad():
                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image_tensor[0],
                        caption=self.target_class,
                        box_threshold=self.BOX_THRESHOLD,
                        text_threshold=self.TEXT_THRESHOLD
                    )

                angle_min = scan_msg.angle_min
                angle_max = scan_msg.angle_max
                angle_inc = scan_msg.angle_increment

                # Procesar detecciones
                for box, phrase, logit in zip(boxes, phrases, logits):
                    confidence = float(logit)
                    if confidence < 0.6:
                        continue

                    # Convertir coordenadas normalizadas a píxeles Y ESCALAR de vuelta al tamaño original
                    cx, cy, bw, bh = box.tolist()
                    
                    # Coordenadas en imagen pequeña
                    small_x1 = int((cx - bw / 2) * small_width)
                    small_x2 = int((cx + bw / 2) * small_width)
                    small_y1 = int((cy - bh / 2) * small_height)
                    small_y2 = int((cy + bh / 2) * small_height)
                    
                    # Escalar coordenadas al tamaño original del frame
                    x1 = int(small_x1 * scale_x)
                    x2 = int(small_x2 * scale_x)
                    y1 = int(small_y1 * scale_y)
                    y2 = int(small_y2 * scale_y)

                    # Verificar que la caja esté dentro de los límites del frame original
                    if not (0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h):
                        continue

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

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
                            my = int((y - self.map_origin.y) / self.map_resolution)

                            if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                                # Agregar detección para el tracker
                                detections_for_tracker.append({
                                    'label': phrase,
                                    'pos': (x, y),
                                    'confidence': confidence
                                })
                                
                                # Agregar detección para el historial (con coordenadas del frame original)
                                frame_detections.append({
                                    'label': phrase,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],  # Coordenadas escaladas al tamaño original
                                    'world_position': {'x': x, 'y': y},
                                    'map_coordinates': {'mx': mx, 'my': my},
                                    'distance': distance,
                                    'angle_deg': math.degrees(angle_rad)
                                })

                                self.last_frame_detections = frame_detections

                        except TransformException as ex:
                            self.get_logger().warn(f"Transform error: {ex}")

            else:
                frame_detections = self.last_frame_detections

                self.last_detection_time = current_time
            # Actualizar tracker
            self.tracker.update(detections_for_tracker, current_time)
            
            # Guardar datos de este frame
            self.save_detection_data(frame_detections, current_time)

            # RENDERIZADO OPTIMIZADO
            if self.matplotlib_initialized and self.fig is not None:
                current_time = time.time()
    
                # Determinar tipo de renderizado
                should_render = self.frame_counter % 5 == 0  # Cada 5 frames
                should_clear = (
                    self.frame_counter % 25 == 0 or  # Cada 25 frames
                    (current_time - getattr(self, 'last_clear_time', 0)) > 3.0  # Cada 3 segundos
                )
                
                if should_render:
                    if should_clear:
                        # Clear completo
                        self.ax_map.clear()
                        self.ax_camera.clear()
                        self.last_clear_time = current_time

                        # Mostrar mapa
                        if self.map_image is not None:
                            self.ax_map.imshow(self.map_image, origin='lower')
                            self.ax_map.set_title("Map View")
                            self.ax_map.set_aspect('equal')
                            
                            # Dibujar robot
                            try:
                                transform = self.tf_buffer.lookup_transform(
                                    self.map_frame,
                                    self.lidar_frame,
                                    rclpy.time.Time(),
                                    timeout=rclpy.duration.Duration(seconds=0.3)
                                )
                                robot_x = transform.transform.translation.x
                                robot_y = transform.transform.translation.y

                                mx = int((robot_x - self.map_origin.x) / self.map_resolution)
                                my = int((robot_y - self.map_origin.y) / self.map_resolution)
                                
                                if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                                    self.ax_map.plot(mx, my, 'mo', markersize=8, markeredgecolor='white', markeredgewidth=2)
                                    
                            except TransformException as ex:
                                pass  # Silenciar warnings para fluidez
                            
                            # Visualizar objetos tracked
                            all_objects = self.tracker.get_all_objects()
                            
                            for track_id, obj in all_objects.items():
                                x, y = obj['pos']
                                mx = int((x - self.map_origin.x) / self.map_resolution)
                                my = int((y - self.map_origin.y) / self.map_resolution)

                                if 0 <= mx < self.map_image.shape[1] and 0 <= my < self.map_image.shape[0]:
                                    color = np.array(self.get_color_for_class(obj['label'])) / 255.0
                                    
                                    # Diferentes estilos para activos/inactivos
                                    if obj['active']:
                                        self.ax_map.plot(mx, my, 'o', color=color, markersize=6)
                                    else:
                                        self.ax_map.plot(mx, my, 'o', color=color, markersize=4, alpha=0.5)
                                    
                                    # Mostrar ID y etiqueta
                                    self.ax_map.text(mx + 5, my + 5, f"{obj['label']}_{track_id}", 
                                                fontsize=8, color=color)

                # Mostrar imagen de cámara con detecciones
                camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.ax_camera.imshow(camera_rgb)
                self.ax_camera.set_title("Camera View")
                self.ax_camera.set_aspect('equal')

                # Dibujar detecciones en la imagen
                for detection in frame_detections:
                    bbox = detection['bbox']
                    label = detection['label']
                    conf = detection['confidence']
                    color = np.array(self.get_color_for_class(label)) / 255.0
                    
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                        edgecolor=color, facecolor='none')
                    self.ax_camera.add_patch(rect)
                    
                    self.ax_camera.text(x1, max(y1 - 10, 0), f"{label} {conf:.2f}", 
                                    fontsize=8, color=color, weight='bold')

                # Crosshair
                h, w = frame.shape[:2]
                self.ax_camera.axvline(x=w//2, color='green', linewidth=1, alpha=0.7)
                self.ax_camera.axhline(y=h//2, color='green', linewidth=1, alpha=0.7)

                # Remover ejes para mejor visualización
                self.ax_map.set_xticks([])
                self.ax_map.set_yticks([])
                self.ax_camera.set_xticks([])
                self.ax_camera.set_yticks([])

                # Información de texto en la figura
                active_objects = self.tracker.get_active_objects()
                info_text = (f"Target: {self.target_class}")

                self.fig.suptitle(f"Map Object Detector - {info_text}")

                # RENDERIZADO MÁS EFICIENTE
                # Solo dibujar cuando hay cambios significativos o cada ciertos frames
                if run_detection or self.frame_counter % 5 == 0:  # Reducir frecuencia de renderizado
                    self.fig.canvas.draw_idle()  # Usar draw_idle en lugar de draw
                    self.fig.canvas.flush_events()

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
        if node.matplotlib_initialized:
            plt.close('all')
    rclpy.shutdown()

if __name__ == '__main__':
    main()