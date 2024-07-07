import gym
import numpy as np
from ZeldaEnv import ZeldaEnv  # Asumiendo que ZeldaEnv está definido en un archivo llamado ZeldaEnv.py
from Mapa import *
from Estado import *
from Utiles import *

def main():
    # Crear una instancia del mapa (usando un archivo de mapa específico, ajusta según sea necesario)
    mapa = Mapa('s_key_0.txt')  # Ajusta el nombre del archivo según tu caso
    
    # Crear una instancia del entorno
    env = ZeldaEnv(task=Task.ZELDA, mapa=mapa)
    
    # Inicializar el entorno
    obs = env.reset()
    
    # Realizar algunos pasos en el entorno
    for _ in range(10):
        action = env.action_space.sample()  # Tomar una acción aleatoria del espacio de acciones
        next_obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Se ha completado la tarea o el jugador ha muerto.")
            break

    env.mapa.mapa[1][1] = CellType.ENEMY
    print(f"Mapa - mapa: {mapa.mapa}")
    print(f"Mapa - env.mapa: {env.mapa.mapa}")
    print(f"Mapa - env.state.mapa: {env.state.mapa.mapa}")

    print("\n")
    print(f"dimension...: {env.observation_space.shape}")
    print(f"size...: {env.obs_size}")
    print(f"np...: {env._get_observation().shape}")
    print("\n")

    # Resetear el entorno y comprobar que todo funcione correctamente
    obs = env.reset()
    print("\nEntorno reseteado.\n")

    print(f"Mapa - mapa: {mapa.mapa}")
    print(f"Mapa - env.mapa: {env.mapa.mapa}")
    print(f"Mapa - env.state.mapa: {env.state.mapa.mapa}")
    print("\n")
    print(f"Score: {env.state.score}")
    print(f"Recompensa: {env.reward}")
    print(f"Posición del jugador: {env.state.posicion_jugador}")
    print(f"Orientación del jugador: {env.state.orientacion_jugador}")
    print(f"Pasos del jugador: {env.state.steps}")
    print(f"Tiene llave: {env.state.tiene_llave}")
    print(f"Vivo: {env.state.alive}")
    print(f"Terminado: {env.state.done}")
    print("\n")
    
    # Realizar algunos pasos más
    for _ in range(5):
        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Se ha completado la tarea o el jugador ha muerto.")
            break
    
    # Cerrar el entorno (opcional)
    env.close()

if __name__ == "__main__":
    main()