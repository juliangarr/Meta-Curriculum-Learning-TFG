# Meta-Curriculum Learning en Juegos de Atari: Algoritmo REPTILE + CL

## Descripción

Este proyecto contiene la implementación explicada en el Trabajo de Fin de Grado titulado "Curriulum Learning en Videojuegos de Atari: desde tareas simples a tareas complejas".


## Estructura del Proyecto

### Carpetas:
- `CSV/`: Contiene archivos CSV con los resultados de evaluación de los modelos.
- `LOGS_SIMPLE/`: Almacena los registros generados durante el entrenamiento de los modelos.
- `Mapas/`: Ficheros .txt de los Mapas utilizados para la versión Simplificada de Zelda.
- `MODELS_SIMPLE/`: Contiene los modelos de RL ya entrenados y utilizados en el proyecto.

### Archivos Python:
- `check_env.py`: Script para verificar el entorno ZeldaEnv.
- `Estado.py`: Define la clase Estado del juego.
- `evaluar_tarea.py`: Contiene la función para la evaluación de tareas específicas.
- `gui.py`: Implementación de la interfaz gráfica de los Mapas.
- `Mapa.py`: Define la clase Mapa del juego.
- `mapa_draw.py`: Fichero para dibujar mapas de ejemplo.
- `prueba_Environment.py`: Fichero para probar el entorno del juego.
- `prueba_Estado.py`: Fichero para probar el funcionamiento del juego.
- `reptile_order.py`: Implementación de Reptile sobre CL alterando el orden.
- `simple.py`: Implementación básica de Curriculum Learning.
- `simple_Reptile.py`: Implementación de Reptile sobre CL.
- `upperbound.py`: Modelo de RL base (baseline).
- `Utiles.py`: Enums utilizados.
- `ZeldaEnv.py`: Definición del entorno de juego.

### Otros Archivos:
-`run.sh`: Script para ejecutar el código de python relevante (resultados).

## Instrucciones de Uso

1. **Preparar el entorno:**
   - Asegúrate de tener Python instalado.
   - Instala las dependencias necesarias (pueden estar listadas en un archivo `requirements.txt` no visible en la captura).

2. **Scripts de prueba:**
   - Para probar el juego desde teclado, ejecuta `python evaluar_tarea.py`
   - Para verificar el entorno, ejecuta `python check_env.py`.
   - Para probar el entorno del juego, ejecuta `python prueba_Environment.py.
   - Para evaluar una tarea, ejecuta `python evaluar_tarea.py`.

3. **Entrenamiento y evaluación de modelos:**
   - Los modelos simples pueden ser entrenados y evaluados utilizando scripts dentro de `MODELS_SIMPLE/`.

