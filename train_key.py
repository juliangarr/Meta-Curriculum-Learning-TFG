import stable_baselines3 as sb3
from evaluar_tarea import evaluar_tarea
from Mapa import *
from ZeldaEnv import *
from Utiles import *

import os
import csv

# Establecer la semilla
SEED = 42

models_dir = "MODELS_prueba_R/KEY"
logs_dir    = "LOGS_prueba_R/KEY"
csv_dir = "CSV"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Tarea a realizar
TASK = Task.FIND_KEY

# Listado de archivos de niveles
level_files = ["key_0.txt", "key_1.txt", "key_2.txt", "key_3.txt", "key_4.txt"]

# Crear los mapas
mapas = [Mapa(f"{file}") for file in level_files]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 100000

# Inicializar el entorno y el modelo
env = ZeldaEnv(mapas[0], TASK, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir}", seed=SEED)

# Entrenar y guardar el modelo para cada mapa
for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
    env = ZeldaEnv(mapa, TASK, pos_jugador=pos)
    model.set_env(env)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Key")
    model.save(f"{models_dir}/model_{i}")
'''
#entrenar la primera vez
env = ZeldaEnv(mapas[0], Task.FIND_KEY, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir}/{0}", seed=SEED)
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Key_{0}")
model.save(f"{models_dir}/model_0")

# Proceder a entrenar el modelo anterior con los demás mapas
for mapa, i, pos in zip(mapas[1:], range(1, len(mapas)), posiciones[1:]):
    env = ZeldaEnv(mapa, Task.FIND_KEY, pos_jugador=pos)
    model = sb3.A2C.load(f"{models_dir}/model_{i-1}", env=env, verbose=1, tensorboard_log=f"{logs_dir}/{i}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Key_{i}")
    model.save(f"{models_dir}/model_{i}")
'''

# Evaluar el modelo en cada uno de los mapas
model_path = f"{models_dir}/model_{len(mapas) - 1}.zip"

# Pedir al usuario el nombre del archivo CSV
csv_filename = "key"

evaluar_tarea(model_path, csv_filename, csv_dir, "KEY", level_files, posiciones, TASK)

