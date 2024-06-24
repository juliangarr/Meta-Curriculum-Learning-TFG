import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation

class SimpleNetwork:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=self.input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(units=self.num_actions, activation='softmax')  # Softmax para clasificación de acciones
        ])
        
        return model
    '''
    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='mean_squared_error')

    def train(self, states, q_values, batch_size=32, epochs=10, validation_split=0.1):
        self.model.fit(states, q_values, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def predict(self, state):
        return self.model.predict(state)

    def summary(self):
        self.model.summary()
    '''

print('Hola mundo')

cnn = SimpleNetwork((123, ), 5)

print('Adios mundo')