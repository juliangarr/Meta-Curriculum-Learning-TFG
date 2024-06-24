from Utiles import *
from Mapa import *
import numpy as np

class Estado:

    def __init__(self, mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador, vivo_jugador = True):
        self.mapa = mapa
        self.orientacion_jugador = orientacion_jugador
        self.posicion_jugador = np.array(posicion_jugador)
        self.steps = pasos_jugador
        self.tiene_llave = llave_jugador
        self.alive = vivo_jugador
    
    def apply_action(self, next_action):
        # Aumentamos en 1 el numero de acciones totales empleadas
        self.steps += 1
        
        if next_action == Action.FORWARD:
            # Modificaciones según nueva posición (antes de cambiar la posicion)
            next_cell = self.get_forward_cell_type()
            if(next_cell == CellType.ENEMY):
                self.alive = False
            elif(next_cell == CellType.KEY):
                self.tiene_llave = True
                key_pos = self.posicion_jugador + self.get_orientation_offset()
                self.mapa.set_cell(key_pos, CellType.FREE)

            # Nueva posición
            self.posicion_jugador = self.get_next_player_position()

        elif next_action == Action.TURN_LEFT:
            self.orientacion_jugador = (self.orientacion_jugador +3) % 4
        
        elif next_action == Action.TURN_RIGHT:
            self.orientacion_jugador = (self.orientacion_jugador +1) % 4
        
        elif next_action == Action.ATACK:
            celda_atacada = self.get_forward_cell_type()

            if celda_atacada == CellType.ENEMY:
                # Modificar el mapa
                next_posicion = self.posicion_jugador + self.get_orientation_offset()
                self.mapa.set_cell(next_posicion, CellType.FREE)
        
        # Matar al jugador si se supera el límite de acciones
        if(self.steps>=500):
            self.alive = False
        
        # siempre devolvemos self modifificado para hacer un uso de memoria eficiente
        return self            

    def get_next_player_position(self): # If we apply MOVE-FORWARD
        next_pos = self.posicion_jugador + self.get_orientation_offset()
        if(self.mapa.get_cell_type(next_pos) != CellType.WALL):
            return next_pos
        else:
            return self.posicion_jugador
                    
    def get_forward_cell_type(self):
        # Obtener el desplazamiento basado en la orientación actual del agente
        next_pos = self.posicion_jugador + self.get_orientation_offset()

        # Obtener el tipo de la celda adyacente
        return self.mapa.get_cell_type(next_pos)
        
    def get_orientation_offset(self):
        moves = {
            0: np.array([-1, 0]),   # UP: Mueve hacia arriba (fila - 1)
            1: np.array([0, 1]),    # RIGHT: Mueve hacia la derecha (columna + 1)
            2: np.array([1, 0]),    # DOWN: Mueve hacia abajo (fila + 1)
            3: np.array([0, -1])    # LEFT: Mueve hacia la izquierda (columna - 1)
        }
        return moves[self.orientacion_jugador]
    
    def is_win(self, task):
        if task == Task.FIND_KEY:
            return self.tiene_llave and self.alive
        
        elif task == Task.FIND_DOOR:
            return self.is_goal()
        
        elif task == Task.KILL_ENEMIES:
            return self.mapa.get_enemies() == 0  and self.alive
        
        elif task == Task.ZELDA:
            return (self.tiene_llave and
                     self.is_goal())
        
        elif task == Task.ALL:
            return (self.tiene_llave and
                     self.is_goal() and
                       self.mapa.get_enemies() == 0)

    def is_goal(self):
        return self.mapa.get_cell_type(self.posicion_jugador) == CellType.DOOR  and self.alive
    
    def flatten_state(self):
        # Flatten the map
        flattened_map = self.mapa.flatten_map()

        # Convert position to a flat array (already flat)
        flat_pos = self.posicion_jugador
        
        # Convert orientation, steps, tiene_llave, and alive to an array
        other_info = np.array([
            self.orientacion_jugador,
            self.steps,
            self.tiene_llave,
            self.alive
        ])
        
        # Concatenate all parts into one vector
        return np.concatenate((flattened_map, flat_pos, other_info))
    
    def get_reward(self, task):
        if self.is_win(task):
            return 1
        elif not self.alive:
            return -1
        else:
            return 0

    def __key(self):
        return (self.posicion_jugador, self.orientacion_jugador, self.steps, self.tiene_llave, self.alive)
    
    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        if isinstance(other, Estado):
            return self.__key() == other.__key()
        return NotImplemented
    

    