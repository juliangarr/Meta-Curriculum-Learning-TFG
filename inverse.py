import stable_baselines3 as sb3
from evaluar_tarea import evaluar_tarea
from Mapa import *
from ZeldaEnv import *
from Utiles import *
import os

# Establecer la semilla
SEED = 42

models_directories = ["MODELS_SIMPLE/INVERSE_ENEMIES", "MODELS_SIMPLE/INVERSE_DOOR", "MODELS_SIMPLE/INVERSE_KEY"]

csv_dir = "CSV"

logs_dir_inicial = "LOGS_SIMPLE"

for models_dir in models_directories:
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

if not os.path.exists(logs_dir_inicial):
    os.makedirs(logs_dir_inicial)

# Tarea a realizar
tasks = [Task.KILL_ENEMIES, Task.FIND_DOOR, Task.FIND_KEY]

# Llave
has_key = [False, True, False]

# Levels names
lvl_names = ["enemies", "door", "key"]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 50000

# Inicializar el entorno y el modelo
mapa_inicial = Mapa("enemies_0.txt")
env = ZeldaEnv(Task.KILL_ENEMIES, mapa_inicial, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir_inicial}", seed=SEED)

# Para cada tarea entrenar todos los mapas
for task, lvl_name, models_dir, llave in zip(tasks, lvl_names, models_directories, has_key):
    # Listado de archivos de niveles
    level_files = [f"{lvl_name}_0.txt", f"{lvl_name}_1.txt", f"{lvl_name}_2.txt", f"{lvl_name}_3.txt", f"{lvl_name}_4.txt"]

    # Crear los mapas
    mapas = [Mapa(f"{file}") for file in level_files]

    # Entrenar y guardar el modelo para cada mapa
    for mapa, i, pos in zip(mapas, range(len(mapas)), posiciones):
        env = ZeldaEnv(task, mapa, pos_jugador=pos, llave_jugador=llave)
        model.set_env(env)
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"INVERSE_{task.name}")
        model.save(f"{models_dir}/model_{i}")

if not os.path.exists("MODELS_SIMPLE/inverse"):
    os.makedirs("MODELS_SIMPLE/inverse")

model.save("MODELS_SIMPLE/inverse/model_final")

# Evaluar el modelo en cada uno de los mapas
model_path = "MODELS_SIMPLE/inverse/model_final.zip"

# Pedir al usuario el nombre del archivo CSV
csv_filename = "inverse"
zelda_levels = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]
evaluar_tarea(model_path, csv_filename, csv_dir, "INVERSE", zelda_levels, posiciones, Task.ZELDA)