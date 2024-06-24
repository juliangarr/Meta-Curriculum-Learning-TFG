import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation

class DeepQNetwork:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        
        # Primera capa convolucional
        model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding='valid', input_shape=self.input_shape))
        model.add(Activation('relu'))
        
        # Segunda capa convolucional
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='valid'))
        model.add(Activation('relu'))
        
        # Aplanar los datos antes de la capa completamente conectada
        model.add(Flatten())
        
        # Capa completamente conectada
        model.add(Dense(256))
        model.add(Activation('relu'))
        
        # Capa de salida completamente conectada (lineal)
        model.add(Dense(self.num_actions))
        
        return model

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='mean_squared_error')

    def train(self, states, q_values, batch_size=32, epochs=10, validation_split=0.1):
        self.model.fit(states, q_values, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, state):
        return self.model.predict(state)

    def summary(self):
        self.model.summary()

# Ejemplo de uso
# input_shape = (input_height, input_width, input_channels) debe ser reemplazado con las dimensiones de la entrada
# num_actions = número de acciones válidas
#input_shape = (84, 84, 4)  # Ejemplo de tamaño de entrada

# COMO MI IMPUT SHAPE SERÁ PLANO, NO TENDRÁ DIMENSIONES DE CANALES
# mi tamaño de entrada es 2 posicion_jugador + 1 orientacion_jugador + 1 pasos_jugador + 1 tiene_llave + 1 alive + filas*columnas = 2 + 1 + 1 + 1 + 1 + 9*13 = 123
tam_vector_entrada = 2 + 1 + 1 + 1 + 1 + 9*13
input_shape = (123, )  # Ejemplo de tamaño de entrada
num_actions = 10  # Ejemplo de número de acciones válidas

dqn = DeepQNetwork(input_shape, num_actions)
dqn.compile(learning_rate=0.001)
dqn.summary()

# Para entrenar el modelo, necesitas proporcionar `states` y `q_values`
# states = np.array([...])  # Debe ser una matriz numpy de estados de entrada
# q_values = np.array([...])  # Debe ser una matriz numpy de valores Q correspondientes
# dqn.train(states, q_values)
