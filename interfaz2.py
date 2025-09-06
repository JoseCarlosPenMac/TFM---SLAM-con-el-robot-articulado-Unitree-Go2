import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import threading
import signal
import os
from PIL import Image, ImageTk

class RobotControlInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Control de Robot - Interfaz de Comandos")
        
        # Configurar pantalla completa según el sistema operativo
        try:
            # Para Windows
            self.root.state('zoomed')
        except:
            try:
                # Para Linux/Mac
                self.root.attributes('-zoomed', True)
            except:
                # Fallback: maximizar manualmente
                self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        
        self.root.configure(bg="#2c3e50")
        
        # Diccionario para almacenar los procesos activos
        self.processes = {
            'launch': None,
            'mover': None,
            'save_map': None,
            'mapeo_v2': None,
            'grounding_dino': None,
            'yolo_tracking': None,    # YOLOv8 Tracking
            'dino_tracking': None     # Grounding DINO Tracking
        }
        
        # Variables para las imágenes
        self.current_detection_image = None
        self.current_pgm_image = None
        
        # Configurar el estilo
        self.setup_styles()
        
        # Crear la interfaz
        self.create_interface()
        
        # Configurar el cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configurar estilos personalizados"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botones de lanzamiento
        style.configure("Launch.TButton",
                       background="#27ae60",
                       foreground="white",
                       font=("Arial", 11, "bold"),
                       padding=8)
        
        # Estilo para botones de detección
        style.configure("Detection.TButton",
                       background="#9b59b6",
                       foreground="white",
                       font=("Arial", 10, "bold"),
                       padding=6)
        
        # Estilo para botones de seguimiento
        style.configure("Tracking.TButton",
                       background="#f39c12",
                       foreground="white",
                       font=("Arial", 10, "bold"),
                       padding=6)
        
        # Estilo para botones de parada individual
        style.configure("Stop.TButton",
                       background="#e74c3c",
                       foreground="white",
                       font=("Arial", 10),
                       padding=6)
        
        # Estilo para botón de parada general
        style.configure("EmergencyStop.TButton",
                       background="#c0392b",
                       foreground="white",
                       font=("Arial", 14, "bold"),
                       padding=15)
        
        # Estilo para botón de cerrar
        style.configure("Close.TButton",
                       background="#34495e",
                       foreground="white",
                       font=("Arial", 12),
                       padding=12)
        
        # Estilo para botones de mapa
        style.configure("Map.TButton",
                       background="#3498db",
                       foreground="white",
                       font=("Arial", 11, "bold"),
                       padding=8)
    
    def create_interface(self):
        """Crear la interfaz gráfica mejorada"""
        # Frame principal con scroll
        main_canvas = tk.Canvas(self.root, bg="#2c3e50")
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas, bg="#2c3e50")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        # Centrar el contenido en el canvas - AMPLIADO
        def center_content(event=None):
            canvas_width = main_canvas.winfo_width()
            content_width = 1600  # Ampliado más para asegurar espacio
            x_offset = max(0, (canvas_width - content_width) // 2)
            main_canvas.coords(canvas_window, x_offset, 0)
        
        canvas_window = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind para recentrar cuando cambie el tamaño
        main_canvas.bind('<Configure>', center_content)
        
        # Título principal
        title_label = tk.Label(scrollable_frame, 
                              text="🤖 CONTROL DE ROBOT UNITREE GO2",
                              font=("Arial", 24, "bold"),
                              fg="#ecf0f1",
                              bg="#2c3e50")
        title_label.pack(pady=20)
        
        # Frame principal con ancho fijo AMPLIADO
        main_frame = tk.Frame(scrollable_frame, bg="#2c3e50", width=1600)  # Ampliado más
        main_frame.pack(pady=10)
        main_frame.pack_propagate(False)
        
        # Configurar grid del frame principal - REAJUSTADO
        main_frame.grid_columnconfigure(0, weight=0, minsize=700)  # Columna izquierda fija más amplia
        main_frame.grid_columnconfigure(1, weight=0, minsize=30)   # Espaciado
        main_frame.grid_columnconfigure(2, weight=0, minsize=870)  # Columna derecha fija más amplia
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Columna izquierda: Todos los controles
        left_frame = tk.Frame(main_frame, bg="#2c3e50", width=700)
        left_frame.grid(row=0, column=0, sticky="nsew", pady=0)
        left_frame.grid_propagate(False)
        
        # Columna derecha: Visualización de mapas AMPLIADA
        right_frame = tk.Frame(main_frame, bg="#2c3e50", width=870)
        right_frame.grid(row=0, column=2, sticky="nsew", pady=0)
        right_frame.grid_propagate(False)
        
        # Crear secciones
        self.create_all_controls_section(left_frame)
        self.create_status_section(left_frame)
        self.create_emergency_section(left_frame)
        self.create_map_section(right_frame)
        self.create_final_buttons(left_frame)
        
        # Configurar el canvas y scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind del scroll del mouse
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Trigger inicial del centrado
        main_canvas.after(100, center_content)
    
    def create_all_controls_section(self, parent):
        """Crear sección unificada de todos los controles"""
        controls_frame = tk.Frame(parent, bg="#34495e", relief="raised", bd=2)
        controls_frame.pack(fill="both", expand=True, pady=10)
        
        title = tk.Label(controls_frame, 
                        text="🎮 CONTROLES DEL ROBOT",
                        font=("Arial", 16, "bold"),
                        fg="#ecf0f1",
                        bg="#34495e")
        title.pack(pady=15)
        
        # Frame principal para organizar controles
        main_controls_frame = tk.Frame(controls_frame, bg="#34495e")
        main_controls_frame.pack(pady=15, padx=15, fill="both", expand=True)
        
        # Configurar grid para 2 columnas
        main_controls_frame.columnconfigure(0, weight=1, minsize=260)
        main_controls_frame.columnconfigure(1, weight=1, minsize=260)
        
        # COLUMNA IZQUIERDA: Controles básicos
        basic_controls_frame = tk.Frame(main_controls_frame, bg="#34495e")
        basic_controls_frame.grid(row=0, column=0, sticky="new", padx=(0, 10), pady=0)
        
        basic_title = tk.Label(basic_controls_frame, 
                              text="📡 BÁSICOS",
                              font=("Arial", 12, "bold"),
                              fg="#ecf0f1",
                              bg="#34495e")
        basic_title.pack(pady=(0, 10))
        
        # Botón 1: Lanzar RVIZ
        btn1_frame = tk.Frame(basic_controls_frame, bg="#34495e")
        btn1_frame.pack(pady=6, fill="x")
        
        ttk.Button(btn1_frame, 
                  text="🚀 Lanzar RVIZ",
                  style="Launch.TButton",
                  command=self.launch_robot_sdk).pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(btn1_frame, 
                  text="Parar",
                  style="Stop.TButton",
                  width=8,
                  command=lambda: self.stop_process('launch')).pack(side="right")
        
        # Botón 2: Mover Robot
        btn2_frame = tk.Frame(basic_controls_frame, bg="#34495e")
        btn2_frame.pack(pady=6, fill="x")
        
        ttk.Button(btn2_frame, 
                  text="🎮 Mover Robot",
                  style="Launch.TButton",
                  command=self.launch_mover_robot).pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(btn2_frame, 
                  text="Parar",
                  style="Stop.TButton",
                  width=8,
                  command=lambda: self.stop_process('mover')).pack(side="right")

        # Botón 3: Guardar Mapa PGM
        btn3_frame = tk.Frame(basic_controls_frame, bg="#34495e")
        btn3_frame.pack(pady=6, fill="x")

        ttk.Button(btn3_frame, 
                  text="💾 Guardar Mapa PGM",
                  style="Launch.TButton",
                  command=self.save_pgm_map).pack(fill="x")
        
        # COLUMNA DERECHA: Controles avanzados CORREGIDA
        advanced_controls_frame = tk.Frame(main_controls_frame, bg="#34495e")
        advanced_controls_frame.grid(row=0, column=1, sticky="new", padx=(0, 10), pady=0)
        
        # Frame contenedor para secciones de detección y seguimiento
        sections_container = tk.Frame(advanced_controls_frame, bg="#34495e")
        sections_container.pack(fill="x", pady=0)
        
        # Configurar grid para dos columnas iguales
        sections_container.columnconfigure(0, weight=1)
        sections_container.columnconfigure(1, weight=1)
    
        # SECCIÓN DETECCIÓN DE OBJETOS (IZQUIERDA)
        detection_section = tk.Frame(sections_container, bg="#34495e")
        detection_section.grid(row=0, column=0, sticky="new", padx=(0, 10), pady=(0,10))
        
        # Etiqueta para Detección de Objetos
        detection_label = tk.Label(detection_section, 
                                 text="🔍 DETECCIÓN DE OBJETOS",
                                 font=("Arial", 12, "bold"),
                                 fg="#ecf0f1",
                                 bg="#34495e")
        detection_label.pack(pady=(0, 10))
        
        # Botones de detección
        # Botón YOLOv8
        ttk.Button(detection_section, 
                text="📊 YOLOv8",
                style="Detection.TButton",
                command=self.launch_mapeo_v2).pack(fill="x", pady=2)
        
        ttk.Button(detection_section, 
                text="Parar y Guardar Mapa YOLOv8",
                style="Stop.TButton",
                command=lambda: self.stop_process('mapeo_v2')).pack(fill="x", pady=2)
        
        # Botón Grounding DINO
        ttk.Button(detection_section, 
                text="🎯 Grounding DINO",
                style="Detection.TButton",
                command=self.launch_grounding_dino).pack(fill="x", pady=2)
        
        ttk.Button(detection_section, 
                text="Parar y Guardar Mapa Grounding DINO",
                style="Stop.TButton",
                command=lambda: self.stop_process('grounding_dino')).pack(fill="x", pady=2)
        
        # SECCIÓN SEGUIMIENTO DE OBJETOS (DERECHA)
        tracking_section = tk.Frame(sections_container, bg="#34495e")
        tracking_section.grid(row=0, column=1, sticky="new", padx=(5, 0))
        
        # Etiqueta para Seguimiento de Objetos
        tracking_label = tk.Label(tracking_section, 
                                text="🎯 SEGUIMIENTO DE OBJETOS",
                                font=("Arial", 12, "bold"),
                                fg="#ecf0f1",
                                bg="#34495e")
        tracking_label.pack(pady=(0, 10))
        
        # Botones de seguimiento
        # Botón YOLOv8 Tracking
        yolo_tracking_frame = tk.Frame(tracking_section, bg="#34495e")
        yolo_tracking_frame.pack(fill="x", pady=2)
        
        ttk.Button(yolo_tracking_frame, 
                  text="📊 YOLOv8",
                  style="Tracking.TButton",
                  command=self.launch_yolo_tracking).pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(yolo_tracking_frame, 
                  text="Parar",
                  style="Stop.TButton",
                  width=8,
                  command=lambda: self.stop_process('yolo_tracking')).pack(side="right")
        
        # Botón Grounding DINO Tracking
        dino_tracking_frame = tk.Frame(tracking_section, bg="#34495e")
        dino_tracking_frame.pack(fill="x", pady=2)
        
        ttk.Button(dino_tracking_frame, 
                  text="🎯 Grounding DINO",
                  style="Tracking.TButton",
                  command=self.launch_dino_tracking).pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        ttk.Button(dino_tracking_frame, 
                  text="Parar",
                  style="Stop.TButton",
                  width=8,
                  command=lambda: self.stop_process('dino_tracking')).pack(side="right")
    
    def create_emergency_section(self, parent):
        """Crear sección de parada general"""
        stop_frame = tk.Frame(parent, bg="#e74c3c", relief="raised", bd=2)
        stop_frame.pack(fill="x", pady=10)
        
        title = tk.Label(stop_frame, 
                        text="🛑 CONTROL DE EMERGENCIA",
                        font=("Arial", 16, "bold"),
                        fg="white",
                        bg="#e74c3c",
                        anchor="center")
        title.pack(pady=15)
        
        # Frame para centrar el botón
        button_frame = tk.Frame(stop_frame, bg="#e74c3c")
        button_frame.pack(expand=True, fill="x")
        
        ttk.Button(button_frame,
                  text="🚨 PARAR TODOS LOS PROCESOS",
                  style="EmergencyStop.TButton",
                  command=self.stop_all_processes).pack(pady=15, expand=True)
    
    def create_status_section(self, parent):
        """Crear sección de estado mejorada"""
        status_frame = tk.Frame(parent, bg="#34495e", relief="raised", bd=2)
        status_frame.pack(fill="x", pady=10)
        
        title = tk.Label(status_frame, 
                        text="📊 ESTADO DE PROCESOS",
                        font=("Arial", 16, "bold"),
                        fg="#ecf0f1",
                        bg="#34495e",
                        anchor="center")
        title.pack(pady=15)
        
        # Frame para indicadores de estado con mejor organización
        indicators_frame = tk.Frame(status_frame, bg="#34495e")
        indicators_frame.pack(fill="x", padx=15, pady=15)
        
        # Crear indicadores para cada proceso (organizados en máximo 2 filas)
        self.status_indicators = {}
        processes = [
            ('launch', '🚀 RVIZ'),
            ('mover', '🎮 Mover Robot'),
            ('save_map', '💾 Guardar Mapa'),
            ('mapeo_v2', '📊 Detección - YOLOv8'),
            ('grounding_dino', '🎯 Detección - Grounding DINO'),
            ('yolo_tracking', '📊 Seguimiento - YOLOv8'),
            ('dino_tracking', '🎯 Seguimiento - Grounding DINO')
        ]
        
        # Organizar en grid con máximo 2 filas (4 elementos en primera fila, 3 en segunda)
        for i, (key, name) in enumerate(processes):
            if i < 4:  # Primera fila
                row = 0
                col = i
            else:  # Segunda fila
                row = 1
                col = i - 4
            
            indicator_frame = tk.Frame(indicators_frame, bg="#2c3e50", relief="sunken", bd=1)
            indicator_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            
            # Configurar columnas para distribución uniforme
            for c in range(4):
                indicators_frame.columnconfigure(c, weight=1)
            
            name_label = tk.Label(indicator_frame, 
                                text=name,
                                font=("Arial", 10, "bold"),
                                fg="#ecf0f1",
                                bg="#2c3e50",
                                anchor="center")
            name_label.pack(pady=3)
            
            status_label = tk.Label(indicator_frame,
                                  text="⚫ Detenido",
                                  font=("Arial", 9),
                                  fg="#95a5a6",
                                  bg="#2c3e50",
                                  anchor="center")
            status_label.pack(pady=3)
            
            self.status_indicators[key] = status_label
    
    def create_map_section(self, parent):
        """Crear sección de visualización de mapas mejorada y EQUILIBRADA"""
        
        # Frame principal contenedor con altura fija
        maps_container = tk.Frame(parent, bg="#2c3e50")
        maps_container.pack(fill="both", expand=True)
        
        # Configurar grid para distribuir equitativamente el espacio
        maps_container.grid_rowconfigure(0, weight=1)  # Mapa de detecciones - 50%
        maps_container.grid_rowconfigure(1, weight=1)  # Mapa PGM - 50%
        maps_container.grid_columnconfigure(0, weight=1)
        
        # Sección superior: Mapa con detecciones (PNG) - ALTURA FIJA
        detection_map_frame = tk.Frame(maps_container, bg="#34495e", relief="raised", bd=2, height=400)
        detection_map_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        detection_map_frame.grid_propagate(False)  # Mantener altura fija
        
        # Título para mapa de detecciones
        detection_title = tk.Label(detection_map_frame, 
                                text="📸 MAPA CON DETECCIONES",
                                font=("Arial", 14, "bold"),
                                fg="#ecf0f1",
                                bg="#34495e",
                                anchor="center")
        detection_title.pack(pady=10)
        
        # Botón para cargar mapa con detecciones
        detection_button_frame = tk.Frame(detection_map_frame, bg="#34495e")
        detection_button_frame.pack(fill="x", padx=15, pady=5)
        
        ttk.Button(detection_button_frame,
                text="📸 Cargar Mapa PNG",
                style="Map.TButton",
                command=self.load_detection_map).pack(expand=True, fill="x")
        
        # Frame para mostrar la imagen de detecciones - ALTURA CONTROLADA
        self.detection_frame = tk.Frame(detection_map_frame, bg="#2c3e50", relief="sunken", bd=1)
        self.detection_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Label para mostrar la imagen de detecciones
        self.detection_label = tk.Label(self.detection_frame, 
                                    text="No hay mapa cargado\n\nUse el botón 'Cargar Mapa PNG'\npara mostrar detecciones",
                                    font=("Arial", 10),
                                    fg="#95a5a6",
                                    bg="#2c3e50")
        self.detection_label.pack(expand=True, fill="both")
        
        # Sección inferior: Mapa PGM - ALTURA FIJA IGUAL
        pgm_map_frame = tk.Frame(maps_container, bg="#34495e", relief="raised", bd=2, height=400)
        pgm_map_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        pgm_map_frame.grid_propagate(False)  # Mantener altura fija
        
        # Título para mapa PGM
        pgm_title = tk.Label(pgm_map_frame, 
                        text="🗺️ MAPA PGM",
                        font=("Arial", 14, "bold"),
                        fg="#ecf0f1",
                        bg="#34495e",
                        anchor="center")
        pgm_title.pack(pady=10)
        
        # Botón para cargar mapa PGM
        pgm_button_frame = tk.Frame(pgm_map_frame, bg="#34495e")
        pgm_button_frame.pack(fill="x", padx=15, pady=5)
        
        ttk.Button(pgm_button_frame,
                text="🗺️ Cargar Mapa PGM",
                style="Map.TButton",
                command=self.load_pgm_map).pack(expand=True, fill="x")
        
        # Frame para mostrar la imagen PGM - ALTURA CONTROLADA IGUAL
        self.pgm_frame = tk.Frame(pgm_map_frame, bg="#2c3e50", relief="sunken", bd=1)
        self.pgm_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Label para mostrar la imagen PGM
        self.pgm_label = tk.Label(self.pgm_frame, 
                                text="No hay mapa cargado\n\nUse el botón 'Cargar Mapa PGM'\npara mostrar el mapa",
                                font=("Arial", 10),
                                fg="#95a5a6",
                                bg="#2c3e50")
        self.pgm_label.pack(expand=True, fill="both")
    
    def create_final_buttons(self, parent):
        """Crear botones finales"""
        final_frame = tk.Frame(parent, bg="#2c3e50")
        final_frame.pack(fill="x", pady=20)
        
        # Frame para centrar el botón
        button_frame = tk.Frame(final_frame, bg="#2c3e50")
        button_frame.pack(expand=True)
        
        ttk.Button(button_frame,
                  text="❌ Cerrar Interfaz",
                  style="Close.TButton",
                  command=self.on_closing).pack()
    
    def load_detection_map(self):
        """Cargar mapa con detecciones (PNG) - TAMAÑO EQUILIBRADO"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Mapa con Detecciones",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Cargar y redimensionar la imagen
                image = Image.open(file_path)
                # Tamaño equilibrado para ambos mapas
                frame_width = 650  
                frame_height = 250  # Reducido para equilibrar con el mapa PGM
                
                # Redimensionar manteniendo la proporción
                image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
                
                # Convertir a PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Actualizar el label
                self.detection_label.configure(image=photo, text="")
                self.detection_label.image = photo  # Mantener referencia
                
                messagebox.showinfo("Éxito", f"Mapa con detecciones cargado:\n{os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el mapa con detecciones:\n{str(e)}")
    
    def run_command(self, command, process_key):
        """Ejecutar comando en un hilo separado"""
        def execute():
            try:
                self.update_process_status(process_key, "🟡 Iniciando...", "#f39c12")
                
                # Ejecutar el comando
                process = subprocess.Popen(command, 
                                         shell=True, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         preexec_fn=os.setsid)
                
                self.processes[process_key] = process
                self.update_process_status(process_key, "🟢 Ejecutando", "#27ae60")
                
                # Esperar a que termine el proceso
                _, stderr = process.communicate()
                
                if process.returncode == 0:
                    self.update_process_status(process_key, "✅ Completado", "#27ae60")
                else:
                    self.update_process_status(process_key, "❌ Error", "#e74c3c")
                
                # Limpiar el proceso del diccionario después de un tiempo
                self.root.after(3000, lambda: self.cleanup_process(process_key))
                
            except Exception as e:
                self.update_process_status(process_key, "❌ Error", "#e74c3c")
                self.processes[process_key] = None
        
        # Ejecutar en un hilo separado
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
    
    def launch_robot_sdk(self):
        """Lanzar RVIZ (anteriormente el SDK del robot)"""
        command = "ros2 launch go2_robot_sdk robot.launch.py"
        self.run_command(command, 'launch')
    
    def launch_mover_robot(self):
        """Lanzar el movimiento del robot"""
        command = "python3 mover_robot_webrtc.py"
        self.run_command(command, 'mover')
    
    def launch_mapeo_v2(self):
        """Lanzar YOLOv8 (anteriormente mapeo v2)"""
        command = "python3 pruebaObjetosSueltos2.py"
        self.run_command(command, 'mapeo_v2')
    
    def launch_grounding_dino(self):
        """Lanzar Grounding DINO v2"""
        command = "python3 pruebaGroundingDino2.py"
        self.run_command(command, 'grounding_dino')
    
    def launch_yolo_tracking(self):
        """Lanzar YOLOv8 Tracking"""
        command = "python3 trackingYolo.py"
        self.run_command(command, 'yolo_tracking')
    
    def launch_dino_tracking(self):
        """Lanzar Grounding DINO Tracking"""
        command = "python3 trackingGroundingDino.py"
        self.run_command(command, 'dino_tracking')
    
    def stop_process(self, process_key):
        """Parar un proceso específico"""
        if self.processes[process_key] is not None:
            try:
                # Terminar el grupo de procesos
                os.killpg(os.getpgid(self.processes[process_key].pid), signal.SIGTERM)
                self.processes[process_key] = None
                self.update_process_status(process_key, "⏹️ Detenido", "#95a5a6")
            except Exception as e:
                pass
        else:
            pass
    
    def stop_all_processes(self):
        """Parar todos los procesos"""
        stopped_count = 0
        for key in self.processes:
            if self.processes[key] is not None:
                try:
                    os.killpg(os.getpgid(self.processes[key].pid), signal.SIGTERM)
                    self.processes[key] = None
                    self.update_process_status(key, "⏹️ Detenido", "#95a5a6")
                    stopped_count += 1
                except Exception as e:
                    pass
        
        if stopped_count > 0:
            messagebox.showinfo("Procesos Detenidos", f"{stopped_count} procesos han sido detenidos")
        else:
            messagebox.showinfo("Sin Procesos", "No hay procesos ejecutándose")
    
    def update_process_status(self, process_key, status_text, color):
        """Actualizar el indicador visual de estado de un proceso"""
        if process_key in self.status_indicators:
            self.status_indicators[process_key].config(text=status_text, fg=color)
    
    def cleanup_process(self, process_key):
        """Limpiar el proceso y resetear el indicador"""
        self.processes[process_key] = None
        self.update_process_status(process_key, "⚫ Detenido", "#95a5a6")
    
    def on_closing(self):
        """Manejar el cierre de la aplicación"""
        if messagebox.askokcancel("Cerrar", "¿Seguro que quieres cerrar la interfaz?"):
            # Detener todos los procesos antes de cerrar
            self.stop_all_processes()
            self.root.destroy()

    def save_pgm_map(self):
        """Guardar mapa PGM"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        map_name = f"/home/josecarlos/ros2_ws/src/go2_robot_sdk/scripts/saved_map_{timestamp}"
        command = f"ros2 run nav2_map_server map_saver_cli -f {map_name}"
        self.run_command(command, 'save_map')

    def load_pgm_map(self):
        """Cargar mapa PGM"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Mapa PGM",
            filetypes=[("PGM files", "*.pgm"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Cargar la imagen PGM
                image = Image.open(file_path)
                
                # Convertir a RGB si es necesario
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Tamaño máximo disponible en el frame
                frame_width = 650
                frame_height = 250
                
                # Primero redimensionar manteniendo proporciones (como antes)
                image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
                
                # Ahora AMPLIAR la imagen un 50% adicional
                current_width, current_height = image.size
                new_width = int(current_width * 2.0)
                new_height = int(current_height * 2.0)
                
                # Redimensionar con el factor de ampliación
                image_enlarged = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Si la imagen ampliada es más grande que el frame, centrarla y recortarla
                if new_width > frame_width or new_height > frame_height:
                    left = max(0, (new_width - frame_width) // 2)
                    top = max(0, (new_height - frame_height) // 2)
                    right = left + min(frame_width, new_width)
                    bottom = top + min(frame_height, new_height)
                    image_final = image_enlarged.crop((left, top, right, bottom))
                else:
                    image_final = image_enlarged
                
                # Convertir a PhotoImage
                photo = ImageTk.PhotoImage(image_final)
                
                # Actualizar el label
                self.pgm_label.configure(image=photo, text="")
                self.pgm_label.image = photo  # Mantener referencia
                
                messagebox.showinfo("Éxito", f"Mapa PGM cargado (ampliado 50%):\n{os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el mapa PGM:\n{str(e)}")
        

def main():
    root = tk.Tk()
    app = RobotControlInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()