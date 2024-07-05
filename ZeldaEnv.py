import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Estado import *
import copy

class ZeldaEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, mapa, task, pos_jugador=[1, 1], orientacion_jugador=1, pasos_jugador=0.0, llave_jugador=False, vivo=True, terminado=False):
        super(ZeldaEnv, self).__init__()
        #super().__init__()

        # Guardar el mapa y la tarea
        self.mapa = mapa
        self.task = task

        # Guardar el mapa, la posición, orientación y llave INICIALES del jugador
        self.pos_inicial = np.array(pos_jugador).copy()
        self.orientacion_inicial = orientacion_jugador
        self.llave_inicial = llave_jugador
        self.mapa_inicial = copy.deepcopy(mapa)

        self.state = Estado(self.mapa, [pos_jugador[0], pos_jugador[1]], orientacion_jugador, pasos_jugador, llave_jugador, vivo, terminado) # Estado inicial, se actualiza en reset()
        
        # Definir las acciones: 0 = FORWARD, 1 = TURN_LEFT, 2 = TURN_RIGHT, 3 = ATTACK
        self.action_space = spaces.Discrete(4)
        
        # Definir el espacio de observación
        self.obs_size = self.mapa.rows * self.mapa.cols + 5  # Tamaño del mapa aplanado
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.obs_size,),  # Mapa aplanado + posición (x, y) + orientación + pasos + tiene_llave
            dtype=np.float32
        )

        # Inicializar la recompensa total
        self.total_reward = 0
        self.pasos = self.state.steps

    def reset(self, seed=None):
        #super().reset(seed=seed)
        self.total_reward = 0
        self.pasos = 0

        # Reiniciar el mapa y el estado del jugador
        self.mapa.mapa = self.mapa_inicial.mapa.copy()
        self.mapa.enemies = self.mapa_inicial.enemies
        self.state = Estado(self.mapa, [self.pos_inicial[0], self.pos_inicial[1]], self.orientacion_inicial, llave_jugador=self.llave_inicial)
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.state.apply_action(action)
        
        # Calcular recompensa
        reward = self.state.get_reward(self.task)
        self.total_reward += reward
        self.pasos += 1

        done = self.state.done or self.state.is_win(self.task)
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, done, False, info

    def _get_observation(self):
        # Obtener el mapa aplanado y normalizado (valores entre 0 y 1)
        flat_map = self.mapa.mapa.flatten() / 4.0  # Dividir por 4 para normalizar si los valores de CellType van de 0 a 4
        
        # Información adicional
        position = self.state.posicion_jugador / np.array([self.mapa.rows, self.mapa.cols])  # Normalizar posición
        orientation = self.state.orientacion_jugador / 3.0  # Normalizar orientación (valores entre 0 y 3)
        steps = np.array([self.state.steps/100.0])
        has_key = np.array([1.0 if self.state.tiene_llave else 0.0])

        # Concatenar toda la información en un solo vector
        return np.concatenate([flat_map, position, [orientation], steps, has_key]).astype(np.float32)

    def _get_info(self):
        return {
            "no info": 0
        }

    def render(self, mode='human'):
        # Aquí podrías implementar una visualización del estado actual si lo deseas
        pass

    def close(self):
        pass