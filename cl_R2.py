import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *
import os
from evaluar_tarea import evaluar_tarea

models_dir = "MODELS_CL/CL_R2"
logs_dir    = "LOGS_CL/CL_R2"
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

TIMESTEPS = 50000

# Inicializar el entorno y el modelo
env = ZeldaEnv(TASK, mapas[0], pos_jugador=posiciones[0])
model = sb3.A2C.load("MODELS_SIMPLE/simple_r2/model_final.zip", env=env, verbose=1, tensorboard_log=f"{logs_dir}")

# Entrenar y guardar el modelo para cada mapa
for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
    env = ZeldaEnv(TASK, mapa, pos_jugador=pos)
    model.set_env(env)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"CL_R2")
    model.save(f"{models_dir}/model_{i}")

# Evaluar el modelo en cada uno de los mapas
model_path = f"{models_dir}/model_{len(mapas) - 1}.zip"
evaluar_tarea(model_path, "cl_R2", csv_dir, "CL_R2", level_files, posiciones, TASK)
