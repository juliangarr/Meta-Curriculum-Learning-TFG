from enum import IntEnum

class Action(IntEnum):
    IDLE = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ATACK = 4

class CellType(IntEnum):
    FREE = 0
    WALL = 1
    ENEMY = 2
    KEY = 3
    DOOR = 4
    #AGENT = 5

class Task(IntEnum):
    FIND_KEY = 0
    FIND_DOOR = 1
    KILL_ENEMIES = 2
    ZELDA = 3
    ALL = 4
'''
class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
'''