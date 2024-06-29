# import tensorflow as tf

# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))

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