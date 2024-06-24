from Utiles import *
import numpy as np

class Mapa:
    def __init__(self, filename):
        self.mapa = self.read_map(filename)
        self.rows = len(self.mapa)
        self.cols = len(self.mapa[0])

    def read_map(self, filename):
        ruta = 'Mapas/' + filename
        
        # Abrir el archivo y leer el contenido
        with open(ruta, 'r') as archivo:
            # Leer todas las líneas y eliminar caracteres de nueva línea
            lineas = [linea.strip() for linea in archivo.readlines()]
        
        # Convertir las líneas en una matriz
        matriz_mapa = []
        for linea in lineas:
            fila = [int(caracter) for caracter in linea]  # Convertir caracteres a enteros
            matriz_mapa.append(fila)
        
        return matriz_mapa

    def get_cell_type(self, position):
        if position[0] < 0 or position[0] >= self.rows or position[1] < 0 or position[1] >= self.cols:
            return CellType.WALL
        else:
            return self.mapa[position[0]][position[1]]
        
    def get_enemies(self):
        enemies = 0
        for row in self.mapa:
            for cell in row:
                if cell == CellType.ENEMY:
                    enemies += 1
        return enemies
    '''
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
    def set_cell(self, position, cell_type):
        self.mapa[position[0]][position[1]] = int(cell_type)

    def flatten_map(self):
        return np.array(self.mapa).flatten()