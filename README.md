# SLAM con el robot articulado Unitree Go2

Este repositorio contiene los *scripts* principales desarrollados en el Trabajo Final de Máster para integrar **navegación autónoma, mapeado y detección o seguimiento de objetos** en el robot Unitree Go2.  

La forma **recomendada** de uso es a través de la **interfaz gráfica**, que lanza y coordina todos los procesos desde un único punto.  

---

## 📂 Preparación del entorno

Para configurar correctamente el proyecto, se deben seguir los siguientes pasos:

1. **Crear el espacio de trabajo (si no existe):**  
   ```bash
   mkdir -p /EspacioDeTrabajo/src
   ```

2. **Clonar el repositorio go2_robot_sdk** (mencionado en la memoria del proyecto):
   ```bash
   cd /EspacioDeTrabajo/src
   git clone [URL_DEL_REPOSITORIO_GO2_ROBOT_SDK]
   ```

3. **Añadir los archivos de este repositorio** en la ruta del SDK:
   ```bash
   # Copiar todos los scripts de este repositorio a:
   /EspacioDeTrabajo/src/go2_robot_sdk/scripts/
   ```

4. **Compilar el espacio de trabajo:**
   ```bash
   cd /EspacioDeTrabajo
   colcon build
   source install/setup.bash
   ```

---

## 🚀 Ejecución del Sistema

### ✅ **Método Recomendado: Interfaz Gráfica**

**La interfaz gráfica es la forma más recomendada de ejecutar el sistema**, ya que permite lanzar y coordinar todos los procesos desde un único punto de control.

```bash
cd /EspacioDeTrabajo/src/go2_robot_sdk/scripts
python3 interfaz2.py
```

Desde la interfaz podrás ejecutar y observar de forma integrada:
- Lanzamiento del robot y sus sensores
- Navegación autónoma
- Guardado de mapas
- Detección de objetos con las redes neuronales YOLOv8 y Grounding DINO
- Seguimiento de objetos específicos usando las redes neuronales YOLOv8 y Grounding DINO
- Carga de mapas en formato .pgm y .png
- Consulta de estado de procesos
- Cierre de emergencia de los procesos

---

## ⚙️ Ejecución Manual de Procesos Individuales

Si necesitas ejecutar los procesos por separado, puedes usar los siguientes comandos:

### 1. **Lanzar el robot y sus sensores**
```bash
ros2 launch go2_robot_sdk robot.launch.py
```

### 2. **Navegación autónoma**
```bash
python3 mover_robot_webrtc.py
```

### 3. **Guardar mapa**
```bash
ros2 run nav2_map_server map_saver_cli
```
*Guarda el mapa generado en formato .pgm y .yaml*

### 4. **Detección de objetos con YOLOv8**
```bash
python3 pruebaObjetosSueltos2.py
```
*Detecta objetos y los marca en el mapa utilizando YOLOv8*

### 5. **Detección de objetos con Grounding DINO**
```bash
python3 pruebaGroundingDino2.py
```
*Detecta objetos y los marca en el mapa utilizando Grounding DINO*

### 6. **Seguimiento de objetos con YOLOv8**
```bash
python3 trackingYolo.py
```
*Detecta y sigue objetos específicos usando YOLOv8*

### 7. **Seguimiento de objetos con Grounding DINO**
```bash
python3 trackingGroundingDino.py
```
*Detecta y sigue objetos específicos usando Grounding DINO*

---

## 📋 Descripción de Funcionalidades

- **🗺️ SLAM:** Implementación de mapeo simultáneo y localización para generar mapas del entorno en tiempo real.
- **🤖 Navegación autónoma:** Sistema de control inteligente que permite al robot moverse de forma autónoma por el entorno mapeado.
- **👁️ Detección de objetos:** Identificación y localización de objetos en el entorno utilizando los modelos YOLOv8 y Grounding DINO.
- **🎯 Seguimiento:** Sistema de tracking que permite al robot seguir objetos específicos detectados usando YOLOv8 y Grounding DINO.
- **💾 Guardado de mapas:** Funcionalidad para guardar manualmente los mapas generados en formatos estándar (.pgm y .yaml).
- **🖥️ Interfaz unificada:** Panel de control centralizado que integra y coordina todas las funcionalidades del sistema.

---

## 📚 Referencias

Para más información sobre la implementación y detalles técnicos, consultar la memoria completa del TFM.

---

## 👨‍💻 Autor

**José Carlos Penalva Maciá**  
**Trabajo Fin de Máster en Inteligencia Artificial**  
**Universidad de Alicante**
