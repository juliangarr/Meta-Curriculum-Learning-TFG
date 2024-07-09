import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *
from evaluar_tarea import evaluar_tarea
import os

# Establecer la semilla
SEED = 42

#models_directories = ["MODELS_SIMPLE/R_ORDER_KEY", "MODELS_SIMPLE/R_ORDER_DOOR", "MODELS_SIMPLE/R_ORDER_ENEMIES"]
#logs_directories    = ["LOGS_SIMPLE/KEY_REPTILE", "LOGS_SIMPLE/DOOR_REPTILE", "LOGS_SIMPLE/ENEMIES_REPTILE"]
csv_dir = "CSV"

models_dir = "MODELS_SIMPLE/R_INVERSE_ORDER"
logs_dir_inicial = "LOGS_SIMPLE/R_INVERSE_ORDER"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

if not os.path.exists(logs_dir_inicial):
    os.makedirs(logs_dir_inicial)

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
tasks = [Task.KILL_ENEMIES, Task.FIND_DOOR, Task.FIND_KEY]

# Levels names
lvl_names = ["enemies", "door", "key"]

# Llave
has_key = [False, True, False]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 50000
ALPHA = 0.1          # Tasa de aprendizaje para la actualización de Reptile

# Inicializar el entorno y el modelo
mapa_inicial = Mapa("enemies_0.txt")
env = ZeldaEnv(Task.KILL_ENEMIES, mapa_inicial, pos_jugador=posiciones[0])
model = sb3.A2C('MlpPolicy', env, verbose=1, tensorboard_log=f"{logs_dir_inicial}", seed=SEED)

# Para cada tarea entrenar todos los mapas
for i, pos  in zip(range(len(posiciones)), posiciones):
    # Listado de archivos de niveles
    level_files = [f"enemies_{i}.txt", f"door_{i}.txt", f"key_{i}.txt"]

    # Crear los mapas
    mapas = [Mapa(f"{file}") for file in level_files]

    # Entrenar y guardar el modelo para cada mapa
    for task, mapa, llave in zip(tasks, mapas, has_key):
        env = ZeldaEnv(task, mapa, pos_jugador=pos, llave_jugador=llave)
        model.set_env(env)

        # Coger los parámetros iniciales
        old_params = copy.deepcopy(model.policy.state_dict())

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"R_INVERSE_ORDER_{i}_{task.name}")
        
        # Extrar los nuevos parámetros después de entrenar en una tarea
        new_params = copy.deepcopy(model.policy.state_dict())
        
        # Meta-actualización
        for param_key in old_params.keys():
            old_params[param_key] += ALPHA * (new_params[param_key] - old_params[param_key])

        # Establecer los parámetros del modelo a la diferencia ajustada
        model.policy.load_state_dict(old_params)   

        model.save(f"{models_dir}/model_{i}_{task.name}")


if not os.path.exists("MODELS_SIMPLE/simple_r_i_order"):
    os.makedirs("MODELS_SIMPLE/simple_r_i_order")

model.save("MODELS_SIMPLE/simple_r_i_order/model_final")


# Evaluar el modelo en cada uno de los mapas
zelda_levels = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]
evaluar_tarea("MODELS_SIMPLE/simple_r_i_order/model_final.zip", "simple_r_i_order", csv_dir, "R_INVERSE_ORDER", zelda_levels, posiciones, Task.ZELDA)
