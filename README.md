# SLAM con el robot articulado Unitree Go2

**Trabajo Fin de Máster en Inteligencia Artificial (Universidad de Alicante)**  
**Autor:** José Carlos Penalva Maciá  
**Título:** *SLAM con el robot articulado Unitree Go2*

Este repositorio contiene los *scripts* principales desarrollados en el TFM para integrar **navegación autónoma, mapeado y detección/seguimiento de objetos** en el robot Unitree Go2.  
La forma **recomendada** de uso es a través de la **interfaz gráfica**, que lanza y coordina todos los procesos desde un único punto.

---

## 📂 Preparación del entorno

Para configurar correctamente el proyecto, se deben seguir los siguientes pasos:

1. **Crear el espacio de trabajo (si no existe):**  
   Ejecutar el siguiente comando para crear la carpeta `EspacioDeTrabajo` con la estructura necesaria:
   ```bash
   mkdir -p /EspacioDeTrabajo/src

2. **Clonar el repositorio go2_robot_sdk mencionado en la memoria dentro de src:**
   Acceder a la carpeta src del espacio de trabajo y clonar el repositorio:
   
   ```bash
   cd /espacioDeTrabajo/src
   git clone <URL_DEL_REPO_GO2_SDK> go2_robot_sdk
