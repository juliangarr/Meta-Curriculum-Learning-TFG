from Mapa import *
from gui import *

# Select the maps
level_files = ["s_zelda_0.txt", "s_key_0.txt", "s_door_0.txt", "s_enemies_0.txt"]

# Create the maps
mapas = []
for file in level_files:
    mapas.append(Mapa(f"{file}"))

# Set the agent position and orientation
agent_position = (1, 1)
agent_orientation = Direction.RIGHT

# Create the GUIS
for mapa in mapas:
    gui = GameGUI(mapa, agent_position, agent_orientation)
    gui.mainloop()