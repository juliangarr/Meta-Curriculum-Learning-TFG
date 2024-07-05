from stable_baselines3.common.env_checker import check_env
from ZeldaEnv import ZeldaEnv
from Mapa import *
from Estado import *
from Utiles import *

# Crear una instancia del mapa (usando un archivo de mapa específico, ajusta según sea necesario)
mapa = Mapa('key_0.txt')  # Ajusta el nombre del archivo según tu caso

# Crear una instancia del entorno
env = ZeldaEnv(mapa=mapa, task=Task.ZELDA, pos_jugador=[1, 1], orientacion_jugador=0, llave_jugador=False)

# It will check your custom environment and output additional warnings if needed
check_env(env)