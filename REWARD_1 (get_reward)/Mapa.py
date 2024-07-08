from Utiles import *
import numpy as np

class Mapa:
    def __init__(self, filename):
        self.name = filename
        self.mapa = self.read_map()
        self.rows, self.cols = self.mapa.shape  # Utilizar .shape para obtener las dimensiones
        self.enemies = self.get_first_enemies()

    def read_map(self):
        ruta = 'Mapas/' + self.name
        
        # Abrir el archivo y leer el contenido
        with open(ruta, 'r') as archivo:
            # Leer todas las líneas y eliminar caracteres de nueva línea
            lineas = [linea.strip() for linea in archivo.readlines()]
        
        # Convertir las líneas en una matriz NumPy
        matriz_mapa = np.zeros((len(lineas), len(lineas[0])), dtype=int)
        for i, linea in enumerate(lineas):
            matriz_mapa[i] = [int(caracter) for caracter in linea]  # Convertir caracteres a enteros
        
        return matriz_mapa

    def get_cell_type(self, position):
        row, col = position
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.mapa[row, col]
        
        return CellType.WALL
        
    def set_cell(self, position, cell_type):
        self.mapa[position[0]][position[1]] = int(cell_type)

    def flatten_map(self):
        return np.array(self.mapa).flatten()
    
        
    def get_first_enemies(self):
        return np.count_nonzero(self.mapa == CellType.ENEMY)
    
    '''
    def get_enemies(self):
        return self.enemies
    
    def get_key(self):
        key = 0
        for row in self.mapa:
            for cell in row:
                if cell == CellType.KEY:
                    key += 1
        return key
    
    def get_door(self):
        door = 0
        for row in self.mapa:
            for cell in row:
                if cell == CellType.DOOR:
                    door += 1
        return door
    '''
    