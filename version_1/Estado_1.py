from Utiles import *
from Mapa import *
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Estado:

    def __init__(self, mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador, vivo_jugador = True):
        # Atributos BASE
        self.mapa = mapa
        self.orientacion_jugador = orientacion_jugador
        self.posicion_jugador = np.array(posicion_jugador)
        self.steps = pasos_jugador
        self.tiene_llave = llave_jugador
        self.alive = vivo_jugador

        # Atributos para el cálculo de recompensas
        self.colision = False
        self.nueva_casilla = False
        self.consigue_llave = False
        self.elimina_enemigo = False

        # Inicializamos la memoria del agente
        self.memoria = np.zeros((self.mapa.rows, self.mapa.cols))

        # Añadimos la posición actual a la memoria
        self.memoria[self.posicion_jugador[0], self.posicion_jugador[1]] = 1

        # Ponemos los limites de la memoria a 1
        self.memoria[0, :] = 1
        self.memoria[-1, :] = 1
        self.memoria[:, 0] = 1
        self.memoria[:, -1] = 1

        # Definimos las categorías del mapa y orientaciones
        self.categorias_mapa = [CellType.FREE, CellType.WALL,  CellType.ENEMY, CellType.KEY, CellType.DOOR]
        self.categorias_orientacion = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        
        # Creamos los OneHotEncoders
        self.map_encoder = OneHotEncoder(categories=[self.categorias_mapa])
        self.orientation_encoder = OneHotEncoder(categories=[self.categorias_orientacion])
        
        # Ajustamos los encoders con las categorías
        self.map_encoder.fit(np.array(self.categorias_mapa).reshape(-1, 1))
        self.orientation_encoder.fit(np.array(self.categorias_orientacion).reshape(-1, 1))
        
        # Inicializamos la representación plana del estado
        #self.estado_flat = self.flatten_state()
        #self._update_state_dict()
    
    def apply_action(self, next_action):
        # Aumentamos en 1 el numero de acciones totales empleadas
        self.steps += 1.0/100.0

        # Reseteamos la colisión
        self.colision = False
        self.consigue_llave = False
        self.elimina_enemigo = False
        self.nueva_casilla = False
        
        if next_action == Action.FORWARD:
            # Modificaciones según nueva posición (antes de cambiar la posicion)
            next_cell = self.get_forward_cell_type()
            if(next_cell == CellType.ENEMY):
                self.alive = False
            elif(next_cell == CellType.KEY):
                self.tiene_llave = True
                self.consigue_llave = True
                key_pos = self.posicion_jugador + self.get_orientation_offset()
                self.mapa.set_cell(key_pos, CellType.FREE)

            # Nueva posición
            self.posicion_jugador = self.get_next_player_position()

        elif next_action == Action.TURN_LEFT:
            self.orientacion_jugador = (self.orientacion_jugador + Direction.LEFT) % 4
        
        elif next_action == Action.TURN_RIGHT:
            self.orientacion_jugador = (self.orientacion_jugador + Direction.RIGHT) % 4
        
        elif next_action == Action.ATACK:
            celda_atacada = self.get_forward_cell_type()

            if celda_atacada == CellType.ENEMY:
                # Modificar el mapa
                next_posicion = self.posicion_jugador + self.get_orientation_offset()
                self.mapa.set_cell(next_posicion, CellType.FREE)
                self.elimina_enemigo = True
        
        # Matar al jugador si se supera el límite de acciones
        if(self.steps>=1.0):
            self.alive = False
        
        # siempre devolvemos self modifificado para hacer un uso de memoria eficiente
        return self            

    def get_next_player_position(self): # If we apply MOVE-FORWARD
        next_pos = self.posicion_jugador + self.get_orientation_offset()
        if(self.mapa.get_cell_type(next_pos) != CellType.WALL):
            # Modificar la memoria
            if(self.memoria[next_pos[0], next_pos[1]] == 0):
                self.memoria[next_pos[0], next_pos[1]] = 1
                self.nueva_casilla = True
            else:
                self.nueva_casilla = False

            return next_pos
        else:
            self.colision = True
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
        # Convertir el mapa en una representación one-hot
        mapa_flat = self.map_encoder.transform(np.array(self.mapa.mapa).reshape(-1, 1)).toarray().flatten()
        
        # Convertir la orientación en una representación one-hot
        one_hot_orientacion = self.orientation_encoder.transform(np.array([[self.orientacion_jugador]])).toarray().flatten()
        
        pos_x, pos_y = self.posicion_jugador/np.linalg.norm(self.posicion_jugador)

        # Crear el vector de estado completo
        estado_flat = np.concatenate([
            mapa_flat, 
            one_hot_orientacion, 
            np.array([pos_x, pos_y]),
            [self.steps], 
            [int(self.tiene_llave)], 
            [int(self.alive)]
        ])
        
        return estado_flat
    '''
    def get_reward(self, task):
        if self.is_win(task):
            return 1
        elif not self.alive:
            return -1
        else:
    '''
    def get_reward(self, task):
        if self.is_win(task):
            return 1000  # Recompensa por completar la misión
        
        elif self.consigue_llave and (task == Task.ZELDA or task == Task.ALL):
            return 500  # Recompensa por conseguir la llave
            
        elif  self.elimina_enemigo and (task == Task.ZELDA or task == Task.ALL or task == Task.KILL_ENEMIES):
            return 250  # Recompensa por eliminar un enemigo
            
        elif not self.alive and self.steps < 0.99:
            return -1000  # Penalización por morir por enemigo
        
        elif self.colision:
            return -100  # Penalización por colisión
        
        elif self.nueva_casilla:
            #return self.steps  # Recompensa por explorar una nueva área
            return 1
        
        else:
            return -self.steps  # Penalización por cada movimiento
            
    def __key(self):
        return (self.posicion_jugador, self.orientacion_jugador, self.steps, self.tiene_llave, self.alive)
    
    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        if isinstance(other, Estado):
            return self.__key() == other.__key()
        return NotImplemented
    

    