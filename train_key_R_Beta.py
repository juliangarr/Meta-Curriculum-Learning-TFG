import stable_baselines3 as sb3
from evaluar_tarea import evaluar_tarea
from Mapa import *
from ZeldaEnv import *
from Utiles import *

import os
import csv
import copy

# Establecer la semilla
SEED = 42

models_dir = "MODELS_prueba_R/R_BETA_KEY"
logs_dir    = "LOGS_prueba_R/R_BETA_KEY"
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
ALPHA = 0.1          # Tasa de aprendizaje para la actualización de Reptile

# inicializar el entorno y el modelo
env = ZeldaEnv(mapas[0], TASK, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir}", seed=SEED)

#print(model.get_parameters())
#initial_params = copy.deepcopy(model.policy.state_dict())
#print(initial_params)

# Meta-entrenamiento con Reptile
for mapa, pos, i in zip(mapas, posiciones, range(len(mapas))):
    env = ZeldaEnv(mapa, TASK, pos_jugador=pos)
    model.set_env(env)

    old_params = copy.deepcopy(model.policy.state_dict())

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"R_Beta_Key")
        
    new_params = copy.deepcopy(model.policy.state_dict())

    # Meta-actualización
    for param_key in old_params.keys():
        old_params[param_key] += (1.0-ALPHA) * old_params[param_key] + ALPHA * (new_params[param_key] - old_params[param_key])

    # Establecer los parámetros del modelo a la diferencia ajustada
    model.policy.load_state_dict(old_params)   
    
    model.save(f"{models_dir}/model_{i}")

#model.save(f"{models_dir}/model_final")

# Evaluar el modelo en cada uno de los mapas
#model_path = f"{models_dir}/model_final.zip"
model_path = f"{models_dir}/model_{len(mapas) - 1}.zip"

# Nombre del archivo CSV
csv_filename = "key_R_Beta"

evaluar_tarea(model_path, csv_filename, csv_dir, "R_BETA_KEY", level_files, posiciones, TASK)