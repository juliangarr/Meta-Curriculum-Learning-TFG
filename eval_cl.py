import stable_baselines3 as sb3
from evaluar_tarea import evaluar_tarea
from Mapa import *
from ZeldaEnv import *
from Utiles import *

models_dir = "MODELS_CL/CL"
csv_dir = "CSV"

# Listado de archivos de niveles
level_files = ["zelda_0.txt", "zelda_1.txt", "zelda_2.txt", "zelda_3.txt", "zelda_4.txt"]

posiciones = [(1, 1), (7, 1), (1, 11), (7, 4), (7, 1)]

# Evaluar el modelo en cada uno de los mapas
model_path = f"{models_dir}/model_0.zip"

# Pedir al usuario el nombre del archivo CSV
csv_filename = "cl_model_0"

evaluar_tarea(model_path, csv_filename, csv_dir, "CL_model_0", level_files, posiciones, Task.ZELDA)