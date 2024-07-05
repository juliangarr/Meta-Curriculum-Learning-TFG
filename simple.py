import stable_baselines3 as sb3
from evaluar_tarea import evaluar_tarea
from Mapa import *
from ZeldaEnv import *
from Utiles import *

import os
import csv

# Establecer la semilla
SEED = 42

models_directories = ["MODELS_SIMPLE/KEY", "MODELS_SIMPLE/DOOR", "MODELS_SIMPLE/ENEMIES"]
#logs_directories    = ["LOGS_SIMPLE/KEY", "LOGS_SIMPLE/DOOR", "LOGS_SIMPLE/ENEMIES"]

csv_dir = "CSV"
'''
for models_dir, logs_dir in zip(models_directories, logs_directories):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
'''
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Tarea a realizar
tasks = [Task.FIND_KEY, Task.FIND_DOOR, Task.KILL_ENEMIES]

# Levels names
lvl_names = ["key", "door", "enemies"]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 100000

logs_dir_inicial = "LOGS_SIMPLE"

# Inicializar el entorno y el modelo
mapa_inicial = Mapa("key_0.txt")
env = ZeldaEnv(mapa_inicial, Task.FIND_KEY, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir_inicial}", seed=SEED)

# Para cada tarea entrenar todos los mapas
for task, lvl_name, models_dir in zip(tasks, lvl_names, models_directories):
    # Listado de archivos de niveles
    level_files = [f"{lvl_name}_0.txt", f"{lvl_name}_1.txt", f"{lvl_name}_2.txt", f"{lvl_name}_3.txt", f"{lvl_name}_4.txt"]

    # Crear los mapas
    mapas = [Mapa(f"{file}") for file in level_files]

    # Entrenar y guardar el modelo para cada mapa
    for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
        env = ZeldaEnv(mapa, task, pos_jugador=pos)
        model.set_env(env)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{task.name}")
        model.save(f"{models_dir}/model_{i}")

if not os.path.exists("MODELS_SIMPLE/simple"):
    os.makedirs("MODELS_SIMPLE/simple")

model.save("MODELS_SIMPLE/simple/model_final")

# Evaluar el modelo en cada uno de los mapas
model_path = "MODELS_SIMPLE/simple/model_final.zip"

# Pedir al usuario el nombre del archivo CSV
csv_filename = "simple"
zelda_levels = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]
evaluar_tarea(model_path, csv_filename, csv_dir, "SIMPLE", zelda_levels, posiciones, Task.ZELDA)