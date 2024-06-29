import tensorflow as tf
import numpy as np
import random

from Simple_Network import *
from Estado import *
from Mapa import *
from Reptile import *
from Utiles import *

# Fijar la semilla del generador de números aleatorios
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# ----------------------- SIMPLE -----------------------
# ------------------------------------------------------
# Definimos las tareas y sus mapas asociados - SIMPLE
tasks_simple = [Task.FIND_KEY, Task.FIND_DOOR, Task.KILL_ENEMIES]
map_files_simple = ['key_lvl0.txt', 'door_lvl0.txt', 'enemies_lvl0.txt']

# Definir para cada par tarea-map su estado inicial
initial_states = []
for task, map_file in zip(tasks_simple, map_files_simple):
    mapa = Mapa(map_file)
    initial_states.append(Estado(mapa, (1, 1), 0, 0, True, True))

# Definimos que NO tiene llave para la tarea de encontrar la llave
initial_states[0].tiene_llave = False

# ---------------------- COMPLEJO ----------------------
# ------------------------------------------------------
# Definimos las tareas y sus mapas asociados - COMPLEJO
task_complex = [Task.ZELDA]
map_files_complex = ['zelda_lvl0.txt']
inicial_complex = []
inicial_complex.append(Estado(Mapa(map_files_complex[0]), (1, 1), 0, 0, False, True))



# Definir dimensiones de los mapas (común en todos)
filas_map = mapa.rows
cols_map = mapa.cols

# Fijar la semilla
seed = 42
set_seed(seed)

# Crear el modelo y el optimizador
model_simple = SimpleNetwork(input_shape=(filas_map * cols_map + 6,), num_actions=5).model
optimizer_simple = tf.keras.optimizers.Adam(learning_rate=0.001)

# Fijar la semilla de nuevo
set_seed(seed)

model_complex = SimpleNetwork(input_shape=(filas_map * cols_map + 6,), num_actions=5).model
optimizer_complex = tf.keras.optimizers.Adam(learning_rate=0.001)

'''
# Comparar los pesos para verificar que son iguales
for layer1, layer2 in zip(model_simple.layers, model_complex.layers):
    weights1 = layer1.get_weights()
    weights2 = layer2.get_weights()
    for w1, w2 in zip(weights1, weights2):
        print(np.allclose(w1, w2)) # Esto debería imprimir True para todas las capas
'''

# Crear la instancia de Reptile - SIMPLE
reptile_simple = Reptile("baseline_Simple", model_simple, optimizer_simple, tasks_simple, initial_states, num_meta_iters=1, num_episodes_per_task=50, alpha=0.001, gamma=0.95)

# Crear la instancia de Reptile - COMPLEJO
reptile_complex = Reptile("baseline_Complex", model_complex, optimizer_complex, task_complex, inicial_complex, num_meta_iters=1, num_episodes_per_task=50, alpha=0.001, gamma=0.95)

# Crear la instancia de Reptile - COMPLEJO
#reptile_complex = Reptile("baseline_Complex", model_complex, optimizer_complex, task_complex, inicial_complex, num_meta_iters=1, num_episodes_per_task=150, alpha=0.001, gamma=0.95)

# Entrenar los modelos
reptile_simple.train_Reptile()
reptile_complex.train_Reptile()

# Definimos los mapas y las tareas de evaluación
# MISMA TAREA
mapa_eval = Mapa('zelda_lvl0.txt')
initial_eval = Estado(mapa_eval, (1,1), 0, 0, False, True)

# Evaluar
reptile_simple.evaluate_task(Task.ZELDA, initial_eval)
reptile_complex.evaluate_task(Task.ZELDA, initial_eval)

# TAREA NUEVA
mapa_eval_2 = Mapa('zelda_lvl1.txt')
initial_eval_2 = Estado(mapa_eval_2, (7,1), 0, 0, False, True)

# Evaluar
reptile_simple.evaluate_task(Task.ZELDA, initial_eval_2)
reptile_complex.evaluate_task(Task.ZELDA, initial_eval_2)

# TAREAS NUEVAS
eval_key = Mapa('key_lvl0.txt')
initial_key = Estado(eval_key, (1,1), 0, 0, False, True)

eval_door = Mapa('door_lvl0.txt')
initial_door = Estado(eval_door, (1,1), 0, 0, True, True)

eval_enemies = Mapa('enemies_lvl0.txt')
initial_enemies = Estado(eval_enemies, (1,1), 0, 0, True, True)

# Evaluar
reptile_simple.evaluate_task(Task.FIND_KEY, initial_key)
reptile_simple.evaluate_task(Task.FIND_DOOR, initial_door)
reptile_simple.evaluate_task(Task.KILL_ENEMIES, initial_enemies)

reptile_complex.evaluate_task(Task.FIND_KEY, initial_key)
reptile_complex.evaluate_task(Task.FIND_DOOR, initial_door)
reptile_complex.evaluate_task(Task.KILL_ENEMIES, initial_enemies)


