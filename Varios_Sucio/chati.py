import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Definir el modelo y el entorno
class Reptile:
    def __init__(self, model, optimizer, num_tasks, num_grad_steps):
        self.model = model
        self.optimizer = optimizer
        self.num_tasks = num_tasks
        self.num_grad_steps = num_grad_steps

    def train_step(self, inputs, labels, rewards, next_inputs):
        # Realizar un paso de gradiente para la tarea actual
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(labels - predictions) * rewards)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Realizar múltiples pasos de gradiente en múltiples tareas
        original_weights = self.model.get_weights()
        for i in range(self.num_tasks):
            task_weights = [original_weights[j].copy() for j in range(len(original_weights))]
            for j in range(self.num_grad_steps):
                task_inputs = inputs
                task_labels = labels
                task_rewards = rewards
                task_next_inputs = next_inputs

                with tf.GradientTape() as tape:
                    predictions = self.model(task_inputs, training=True)
                    loss = tf.reduce_mean(tf.square(task_labels - predictions) * task_rewards)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                task_weights = [task_weights[k] - 0.1 * gradients[k].numpy() for k in range(len(gradients))]
                self.model.set_weights(task_weights)

        # Resetear los pesos del modelo a los pesos originales
        self.model.set_weights(original_weights)

def create_task_environment(task, map_file):
    mapa = Mapa(map_file)
    posicion_jugador = (3, 3)  # ejemplo
    orientacion_jugador = 0  # ejemplo
    pasos_jugador = 0
    llave_jugador = False
    
    estado = Estado(mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador)
    return estado

def evaluate_state(state, task):
    return state.is_win(task)

def get_state_representation(state):
    state_representation = state.mapa.mapa.copy()
    x, y = state.posicion_jugador
    state_representation[x][y] = 5  # Representar al jugador con un número especial, por ejemplo
    return np.expand_dims(state_representation, axis=-1)

def get_reward(state, task):
    return 1.0 if evaluate_state(state, task) else 0.0

# Define the neural network model with convolutional layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(13, 13, 1)),  # Ajusta el input_shape según tu mapa
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(Action))  # Número de acciones posibles
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
reptile = Reptile(model, optimizer, num_tasks=10, num_grad_steps=10)

NUM_TASKS = 10
NUM_EPISODES_PER_TASK = 10

tasks = [Task.FIND_KEY, Task.FIND_DOOR, Task.KILL_ENEMIES]
map_files = ['map1.txt', 'map2.txt', 'map3.txt']

for i in range(NUM_TASKS):
    for task, map_file in zip(tasks, map_files):
        for j in range(NUM_EPISODES_PER_TASK):
            state = create_task_environment(task, map_file)
            done = False
            while not done:
                state_representation = get_state_representation(state)
                state_representation = np.expand_dims(state_representation, axis=0)
                q_values = model.predict(state_representation)
                action = np.argmax(q_values)

                next_state = state.apply_action(action)
                reward = get_reward(next_state, task)
                reptile.train_step(state_representation,
                                   np.array([action]),
                                   np.array([reward]),
                                   np.expand_dims(get_state_representation(next_state), axis=0))
                
                state = next_state
                done = not state.alive or evaluate_state(state, task)

            # Print total reward for the episode
            print(f"Task: {task.name}, Episode: {j}, Reward: {reward}")

# Evaluación en la tarea ZELDA
zelda_state = create_task_environment(Task.ZELDA, 'zelda_map.txt')

for _ in range(100):  # número de pasos de la simulación
    state_representation = get_state_representation(zelda_state)
    state_representation = np.expand_dims(state_representation, axis=0)
    q_values = model.predict(state_representation)
    action = np.argmax(q_values)
    zelda_state = zelda_state.apply_action(action)

    if evaluate_state(zelda_state, Task.ZELDA):
        print("Success on ZELDA task!")
        break
    else:
        print("Failed to complete ZELDA task.")
