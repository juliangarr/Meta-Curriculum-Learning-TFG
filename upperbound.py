import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *
from evaluar_tarea import evaluar_tarea

import os
import csv

# Establecer la semilla
SEED = 42

models_dir = "MODELS_SIMPLE/UPPERBOUND"
logs_dir    = "LOGS_SIMPLE/UPPERBOUND"
csv_dir = "CSV"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Tarea a realizar
TASK = Task.ZELDA

# Listado de archivos de niveles
level_files = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]

# Crear los mapas
mapas = [Mapa(f"{file}") for file in level_files]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 100000

# Inicializar el entorno y el modelo
env = ZeldaEnv(mapas[0], TASK, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir}", seed=SEED)

# Entrenar 3 veces todos los mapas
for j in range(3):
    mapas = [Mapa(f"{file}") for file in level_files]
    # Entrenar y guardar el modelo para cada mapa
    for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
        env = ZeldaEnv(mapa, TASK, pos_jugador=pos)
        model.set_env(env)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Upperbound")
        model.save(f"{models_dir}/model_{j}_{i}")


# Evaluar el modelo en cada uno de los mapas
model_path = f"{models_dir}/model_2_{len(mapas) - 1}.zip"
csv_file = "evaluate_upperbound_simple"

evaluar_tarea(model_path, csv_file, csv_dir, "UPPERBOUND", level_files, posiciones, TASK)


