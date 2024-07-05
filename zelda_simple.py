import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *
from evaluar_tarea import evaluar_tarea

import os

models_dir  = "models_Simple_Zelda/ZELDA_simple"
logs_dir    = "logs_Simple_Zelda/ZELDA_simple"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Crear el entorno
mapa = Mapa("s_zelda_0.txt")
env = ZeldaEnv(mapa, Task.ZELDA) 

# Indicar la semilla
SEED = 42

# Crear el modelo A2C
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=logs_dir, seed=SEED)

TIMESTEPS = 10
for i in range(1,11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ZELDA_simple")
    model.save(f"{models_dir}/model_{i}")

model_path = f"{models_dir}/model_10"
csv_dir = "CSV"
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

lvl_names = ['s_zelda_0.txt', 's_zelda_1.txt']
csv_name = "zelda_simple.csv"
csv_dir = "CSV"
model_path = f"{models_dir}/model_10.zip"
posiciones = [(1, 1), (1, 1)]
evaluar_tarea(model_path, csv_name, csv_dir, "Z_SIMPLE", lvl_names, posiciones, Task.ZELDA)