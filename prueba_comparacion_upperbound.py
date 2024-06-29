import tensorflow as tf
import numpy as np
import random

from Simple_Network import *
from Estado import *
from Mapa import *
from Reptile import *
from Utiles import *

#crear tarea
mapa_eval = Mapa('find_zelda_0.txt')
initial_eval = Estado(mapa_eval, (1,1), 0, 0, False, True)

#importar modelo
optimizer_simple = tf.keras.optimizers.Adam(learning_rate=0.001)
reptile_simple = Reptile('upperbound', None, optimizer_simple, [], [])
reptile_simple.load_model('06_26_17_57_reducido_Simple')

#evaluar tarea con modelo importado 50 veces
for i in range(50):
    reptile_simple.evaluate_task(Task.ZELDA, initial_eval)