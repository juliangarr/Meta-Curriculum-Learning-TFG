import stable_baselines3 as sb3
from Mapa import *
from ZeldaEnv import *
from Utiles import *
from evaluar_tarea import evaluar_tarea
import os

# Establecer la semilla
SEED = 42

models_directories = ["MODELS_SIMPLE/R_BETA_KEY", "MODELS_SIMPLE/R_BETA_DOOR", "MODELS_SIMPLE/R_BETA_ENEMIES"]

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
tasks = [Task.FIND_KEY, Task.FIND_DOOR, Task.KILL_ENEMIES]

# Levels names
lvl_names = ["key", "door", "enemies"]

# Llave
has_key = [False, True, False]

# Indicar las posiciones iniciales del jugador
posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

TIMESTEPS = 100000
ALPHA = 0.1          # Tasa de aprendizaje para la actualización de Reptile

# Inicializar el entorno y el modelo
mapa_inicial = Mapa("key_0.txt")
env = ZeldaEnv(Task.FIND_KEY, mapa_inicial, pos_jugador=posiciones[0])
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

        # Coger los parámetros iniciales
        old_params = copy.deepcopy(model.policy.state_dict())

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"R_BETA_{task.name}")
        
        # Extrar los nuevos parámetros después de entrenar en una tarea
        new_params = copy.deepcopy(model.policy.state_dict())
        
        # Meta-actualización
        for param_key in old_params.keys():
            old_params[param_key] = (1.0-ALPHA) * old_params[param_key] + ALPHA * (new_params[param_key] - old_params[param_key])

        # Establecer los parámetros del modelo a la diferencia ajustada
        model.policy.load_state_dict(old_params)   

        model.save(f"{models_dir}/model_{i}")


if not os.path.exists("MODELS_SIMPLE/simple_r_beta"):
    os.makedirs("MODELS_SIMPLE/simple_r_beta")

model.save("MODELS_SIMPLE/simple_r_beta/model_final")


# Evaluar el modelo en cada uno de los mapas
zelda_levels = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]
evaluar_tarea("MODELS_SIMPLE/simple_r_beta/model_final.zip", "simple_r_beta", csv_dir, "R_BETA", zelda_levels, posiciones, Task.ZELDA)
