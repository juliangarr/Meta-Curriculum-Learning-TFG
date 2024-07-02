from Estado import *
import numpy as np
import tensorflow as tf
import copy
import os
import csv
from datetime import datetime
import sys

# Definir el modelo y el entorno
class Reptile:
    def __init__(self, name, model, optimizer, tasks, task_initial_states, num_meta_iters = 1, num_episodes_per_task = 1000, alpha = 0.001, gamma=0.95):
        # SET NAME
        now = datetime.now()
        timestamp = now.strftime("%m_%d_%H_%M")
        self.name = f"{timestamp}_{name}"

        # SET ATTRIBUTES
        self.model = model
        self.optimizer = optimizer
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.initial_states = task_initial_states
        self.meta_iterations = num_meta_iters
        self.num_episodes_per_task = num_episodes_per_task
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.history = []

    def train_one_task(self, task, estado_inicial, num_episodes, gamma):
        task_history = {
            "task_name": task.name,
            #"initial_state": estado_inicial.flatten_state().tolist(),
            "initial_state": estado_inicial.mapa.name,
            "successes": 0,
            "failures": 0,
            "episodes": []
        }
        
        for episode in range(num_episodes):
            #estado = estado_inicial
            estado = copy.deepcopy(estado_inicial)
            done = False
            total_reward = 0
            
            while not done:
                # Obtener la acción del modelo actual
                estado_actual = estado.flatten_state()
                
                # Obtener las probabilidades de las acciones del modelo
                acciones_probabilidades = self.model.predict(np.array([estado_actual]))
                #acciones_probabilidades = self.silent_predict(self.model, np.array([estado_actual]))
                accion_elegida = np.argmax(acciones_probabilidades) # seleccionar el índice de la acción con mayor probabilidad
                
                # Aplicar la acción al entorno
                nuevo_estado = estado.apply_action(accion_elegida)
                reward = nuevo_estado.get_reward(task)
                total_reward += reward
                
                # Actualizar el modelo basado en la recompensa obtenida
                with tf.GradientTape() as tape:
                    estado_actual_tensor = tf.convert_to_tensor([estado_actual], dtype=tf.float32)
                    nuevo_estado_tensor = tf.convert_to_tensor([nuevo_estado.flatten_state()], dtype=tf.float32)
                    
                    q_values_actuales = self.model(estado_actual_tensor)
                    q_values_siguientes = self.model(nuevo_estado_tensor)

                    target_q = reward + gamma * np.max(q_values_siguientes)
                    target_q_values = q_values_actuales.numpy()
                    target_q_values[0][accion_elegida] = target_q
                    '''
                    target = reward + gamma * np.max(self.model.predict(nuevo_estado_tensor))
                    #target = reward + gamma * np.max(self.silent_predict(self.model, nuevo_estado_tensor))
                    predicted = self.model(estado_actual_tensor)
                    
                    # Ajusta target para que tenga la misma forma que predicted
                    target_tensor = tf.convert_to_tensor([[target] * predicted.shape[-1]], dtype=tf.float32)
                    #print("Target: ", target_tensor.shape)
                    #print("Predicted: ", predicted.shape)
                    '''
                    loss = tf.keras.losses.MeanSquaredError()(target_q_values, q_values_actuales)
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                estado = nuevo_estado
                done = estado.is_win(task) or not estado.alive
            
            # Guardar el resultado del episodio
            if estado.alive:
                result = 1
                task_history["successes"] += 1
            else:
                result = -1
                task_history["failures"] += 1
            
            episode_history = {
                "episode_index": episode,
                "steps": estado.steps,  # Asumiendo que steps es un atributo de estado
                "result": result,
                "total_reward": total_reward
            }
            task_history["episodes"].append(episode_history)

        # Guardar el historial de la tarea
        self.history.append(task_history)

    def train_Reptile(self):
        alpha = self.learning_rate
        meta_iterations = self.meta_iterations

        # Obtener los parámetros iniciales del modelo
        meta_parameters = self.model.get_weights()

        for iteration in range(meta_iterations):
            for task, initial_state in zip(self.tasks, self.initial_states):
                # Hacer una copia de los parámetros meta actuales
                original_parameters = copy.deepcopy(meta_parameters)
                
                # Entrenar el modelo en la tarea específica
                self.train_one_task(task, initial_state, self.num_episodes_per_task, self.discount_factor)
                
                # Obtener los nuevos parámetros después del entrenamiento
                new_parameters = self.model.get_weights()
                
                # Calcular la diferencia y actualizar los meta-parámetros
                gradient = [(new_p - orig_p) * alpha for new_p, orig_p in zip(new_parameters, original_parameters)]
                meta_parameters = [orig_p + grad for orig_p, grad in zip(original_parameters, gradient)]

        # Establecer los parámetros finales del modelo después del entrenamiento Reptile
        self.model.set_weights(meta_parameters)

        # Guardar el modelo entrenado
        self.save_model()

        # Imprimir el historial de entrenamiento
        self.print_Train_to_csv()

    def evaluate_task(self, e_task, e_state):
        estado = copy.deepcopy(e_state)
        done = False
        
        while not done:
            # Obtener la acción del modelo actual
            estado_actual = estado.flatten_state()
            #acciones_probabilidades = self.model.predict(np.array([estado_actual]))
            acciones_probabilidades = self.silent_predict(self.model, np.array([estado_actual]))
            accion_elegida = np.argmax(acciones_probabilidades)
            
            # Aplicar la acción al entorno
            estado = estado.apply_action(accion_elegida)        
            
            # Actualizar condición de parada
            done = estado.is_win(e_task) or not estado.alive
        
        # Guardar el resultado de la evaluación
        result = 1 if estado.is_win(e_task) else -1
        evaluation_result = {
            "model_name": self.name,
            "task_name": e_task.name,
            "steps": estado.steps,
            "result": result
        }

        # Mostrar resultado
        if  estado.is_win(e_task):
            print("Success on task " + self.name + " - " + e_task.name + "------   KKKKKKK  1")
        else:
            print("Failed to complete task " + self.name + " - " + e_task.name + "------     -1")

        # Imprimir en CSV el resultado de la evaluación
        self.print_Eval_to_csv(evaluation_result)

    def save_model(self):
        directory = "TRAINED_MODELS"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, self.name + ".keras")
        self.model.save(filepath)

    def load_model(self, filename):
        directory = "TRAINED_MODELS"
        if not filename.endswith('.keras'):
            filename += '.keras'
        filepath = os.path.join(directory, filename)
        self.model = tf.keras.models.load_model(filepath)

    def print_Train_to_csv(self):
        filename = f"results_TRAIN/{self.name}.csv"
        if not os.path.exists("results_TRAIN"):
            os.makedirs("results_TRAIN")
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Parameter", "Value"
            ])
            writer.writerow(["Learning rate (alpha)", self.learning_rate])
            writer.writerow(["Discount factor (gamma)", self.discount_factor])
            writer.writerow(["Number of meta iterations", self.meta_iterations])
            writer.writerow(["Number of tasks", self.num_tasks])
            writer.writerow(["Number of episodes per task", self.num_episodes_per_task])
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])

            for task_data in self.history:
                writer.writerow(["Task name", task_data["task_name"]])
                writer.writerow(["Initial state", task_data["initial_state"]])
                writer.writerow(["Number of successes", task_data["successes"]])
                writer.writerow(["Number of failures", task_data["failures"]])
                writer.writerow([])

                writer.writerow([
                    "Episode index", "Steps", "Result", "Total reward"
                ])
                for episode in task_data["episodes"]:
                    writer.writerow([
                        episode["episode_index"],
                        episode["steps"],
                        episode["result"],
                        episode["total_reward"]
                    ])
                writer.writerow([])  # Blank line between tasks
                writer.writerow([])
                writer.writerow([])

    def print_Eval_to_csv(self, evaluation_result):
        task_name = evaluation_result["task_name"]
        filename = f"results_EVAL/eval_{self.name}_{task_name}.csv"
        if not os.path.exists("results_EVAL"):
            os.makedirs("results_EVAL")
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Model Name", "Task Name", "Steps", "Result"
            ])
            writer.writerow([
                evaluation_result["model_name"],
                evaluation_result["task_name"],
                evaluation_result["steps"],
                evaluation_result["result"]
            ])

    def silent_predict(self, model, data):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = open(os.devnull, 'w')  # Redirect standard output to devnull
        result = model.predict(data)
        sys.stdout.close()
        sys.stdout = original_stdout  # Reset the standard output to its original value
        return result
    
    '''
    def silent_predict(self, model, data):
        with open(os.devnull, 'w') as fnull:
            original_stdout = sys.stdout
            sys.stdout = fnull
            try:
                result = model.predict(data)
            finally:
                sys.stdout = original_stdout
        return result
    '''