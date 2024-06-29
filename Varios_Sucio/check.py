# import tensorflow as tf

# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
'''
import timeit

# Definir las operaciones
def multiplicacion():
    return 0 * 0.25

def division():
    return 0 / 4

# Medir el tiempo de ejecución de cada operación
tiempo_multiplicacion = timeit.timeit(multiplicacion, number=1000000)
tiempo_division = timeit.timeit(division, number=1000000)

print(f"Tiempo de multiplicación: {tiempo_multiplicacion} segundos")
print(f"Tiempo de división: {tiempo_division} segundos")

if tiempo_multiplicacion < tiempo_division:
    print("La multiplicación es más rápida")
else:
    print("La división es más rápida")
    '''

import numpy as np

# Ejemplo de normalización por la norma L2 para las posiciones del jugador
posicion_jugador = np.array([5, 7])  # Ejemplo de posición del jugador

norma = np.linalg.norm(posicion_jugador)  # Calculamos la norma L2 del vector

posicion_jugador_norm = posicion_jugador / norma

print("Posición del jugador normalizada (norma L2):", posicion_jugador_norm)