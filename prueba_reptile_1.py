from Simple_Network import *
from Estado import *
from Mapa import *
from Reptile import *
from Utiles import *

# Definimos las tareas y sus mapas asociados
tasks = [Task.FIND_KEY, Task.FIND_DOOR, Task.KILL_ENEMIES]
map_files = ['find_key_0.txt', 'find_door_0.txt', 'kill_enemies_0.txt']

# Definir para cada par tarea-map su estado inicial
initial_states = []
for task, map_file in zip(tasks, map_files):
    mapa = Mapa(map_file)
    initial_states.append(Estado(mapa, (1, 1), 0, 0, True, True))

# Definir dimensiones de los mapas (común en todos)
filas_map = mapa.rows
cols_map = mapa.cols

# Definimos que NO tiene llave para la tarea de encontrar la llave
initial_states[0].tiene_llave = False

'''
# Definimos los estados iniciales planos
initial_states_flatten = []
for estado in initial_states:
    initial_states_flatten.append(estado.flatten_state())
'''

# Crear el modelo y el optimizador
cnn = SimpleNetwork(input_shape=(filas_map * cols_map + 6,), num_actions=5)
model = cnn.model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Crear la instancia de Reptile
reptile = Reptile("prueba_simple_2", model, optimizer, tasks, initial_states, num_meta_iters=1, num_episodes_per_task=5, alpha=0.001, gamma=0.95)

# Entrenar el modelo
reptile.train_Reptile()

# Definimos el mapa y la tarea de evaluación
mapa_eval = Mapa('eval_key_0.txt')
initial_eval = Estado(mapa_eval, (4,5), 0, 0, 0, True)

# Evaluar una tarea
reptile.evaluate_task(Task.FIND_KEY, initial_eval)