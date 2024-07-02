import gym
from gym import spaces
import numpy as np
import Estado
from Utiles import *

class ZeldaEnv(gym.Env):
    def __init__(self, mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador, task):
        super(ZeldaEnv, self).__init__()
        
        # Inicializamos el estado
        self.estado = Estado(mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador)
        self.task = task
        
        # Definimos la acción y el espacio de observación
        self.action_space = spaces.Discrete(4)  # FORWARD, TURN_LEFT, TURN_RIGHT, ATACK
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.estado.flatten_state()),), dtype=np.float32)
    
    def reset(self):
        # Resetear el entorno a su estado inicial y devolver la primera observación
        self.estado = Estado(self.mapa, self.posicion_jugador, self.orientacion_jugador, self.pasos_jugador, self.llave_jugador)
        return self.estado.flatten_state()
    
    def step(self, action):
        # Aplicar la acción y obtener el nuevo estado
        next_action = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.ATACK][action]
        self.estado = self.estado.apply_action(next_action)
        
        # Obtener la nueva observación, recompensa y si el episodio ha terminado
        obs = self.estado.flatten_state()
        reward = self.estado.get_reward(self.task)
        done = not self.estado.alive or self.estado.is_win(self.task)
        
        return obs, reward, done, {}

    def render(self, mode='human'):
        # Opcional: Implementar la visualización del entorno
        pass
