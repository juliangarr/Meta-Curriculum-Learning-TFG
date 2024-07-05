# Asegúrate de que estás en el directorio correcto para acceder a los archivos
# Por si ejecutas el script prueba_Estado.py desde un directorio diferente al que contiene estos módulos.
# sys.path.append(os.path.abspath('.'))

from Utiles import *
from Mapa import *
from Estado import *

# Definir un mapa de ejemplo para inicialización
mapa = Mapa('zelda_0.txt')

# Posición inicial del jugador y orientación (por ejemplo, arriba (0))
posicion_jugador = (1, 1)  # Cambia esta posición según el mapa
orientacion_jugador = 0  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
pasos_jugador = 0.0
llave_jugador = False
vivo_jugador = True

estado = Estado(mapa, posicion_jugador, orientacion_jugador, pasos_jugador, llave_jugador, vivo_jugador)

# Función para imprimir el estado actual del mapa y jugador
def print_estado(estado):
    #os.system('clear')  # Limpiar la pantalla para Unix/Linux/Mac
    for i in range(estado.mapa.rows):
        for j in range(estado.mapa.cols):
            if (i, j) == tuple(estado.posicion_jugador):
                print('J', end=' ')
            else:
                print(estado.mapa.get_cell_type((i, j)), end=' ')
        print()
    print(f"Posición del jugador: {estado.posicion_jugador}")
    print(f"Orientación del jugador: {estado.orientacion_jugador}")
    print(f"Pasos del jugador: {estado.steps}")
    print(f"Jugador tiene llave: {estado.tiene_llave}")
    print(f"Jugador está vivo: {estado.alive}")

    print('\n')
    print(f"Colisión: {estado.colision}")
    print(f"Consigue llave: {estado.consigue_llave}")
    print(f"Elimina enemigo: {estado.elimina_enemigo}")
    print(f"Nueva casilla: {estado.nueva_casilla}")
    print(f"Terminado: {estado.done}")
    print(f"is_GOAL: {estado.is_goal()}")
    print('\n')
    print(f"Memoria: \n")
    print(estado.memoria)

    #print('\n')
    #print(estado.flatten_state())
    print('\n')
    #print(estado.flatten_state().shape)
    print('\n')


# Función para convertir teclas en acciones
def get_action_from_key(key):
    if key == 'w':
        return Action.FORWARD
    elif key == 'a':
        return Action.TURN_LEFT
    elif key == 'd':
        return Action.TURN_RIGHT
    elif key == 'p':
        return Action.ATACK
    #else:
    #    return Action.IDLE

# Bucle principal del juego
while not estado.done and not estado.is_win(Task.ZELDA):
    print_estado(estado)
    key = input("Introduce una acción (W/A/S/D/P): ").lower()
    action = get_action_from_key(key)
    estado = estado.apply_action(action)

if estado.done:
    if estado.alive:
        print("¡Has perdido!")
    else:
        print("¡Has muerto!")
    
else:
    print("¡Has ganado!")