from stable_baselines3.common.env_checker import check_env
from ZeldaEnv import ZeldaEnv
from Mapa import *
from Estado import *
from Utiles import *

# Crear una instancia del mapa (usando un archivo de mapa específico, ajusta según sea necesario)
mapa = Mapa('key_0.txt')  # Ajusta el nombre del archivo según tu caso

# Crear una instancia del entorno
env = ZeldaEnv(task=Task.FIND_KEY, mapa=mapa)

# It will check your custom environment and output additional warnings if needed
check_env(env)