import tkinter as tk

from Utiles import CellType, Direction

class GameGUI(tk.Tk):
    def __init__(self, mapa, agent_position, agent_orientation):
        super().__init__()
        
        self.title("Game GUI")
        
        self.rows = mapa.rows
        self.cols = mapa.cols

        self.create_gui(mapa, agent_position, agent_orientation)
     
    def create_gui(self, game_map, agent_position, agent_orientation):
        self.canvas = tk.Canvas(self, width = self.cols * 30, height = self.rows * 30)
        self.canvas.pack()
        
        self.update_gui(game_map, agent_position, agent_orientation)

    def update_gui(self, game_map, agent_position, agent_orientation):
        self.canvas.delete("all")

        for i in range(self.rows):
            for j in range(self.cols):
                cell_type = game_map.mapa[i][j]
                color = self.get_color(cell_type)

                x1, y1 = j * 30, i * 30
                x2, y2 = x1 + 30, y1 + 30

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

        # Draw the agent (triangle)
        agent_x, agent_y = agent_position
        self.draw_triangle(agent_y, agent_x, agent_orientation)
        
        self.update()

    def draw_triangle(self, x, y, orientation):
        length = 20
        width = 10

        direction = {
            Direction.DOWN: (0, 1),
            Direction.RIGHT: (1, 0),
            Direction.UP: (0, -1),
            Direction.LEFT: (-1, 0)
        }[orientation]

        # Compute the triangle vertices
        x1 = x * 30 + 15 - width * direction[1]
        y1 = y * 30 + 15 + width * direction[0]
        x2 = x * 30 + 15 + length * direction[0]
        y2 = y * 30 + 15 + length * direction[1]
        x3 = x * 30 + 15 + width * direction[1]
        y3 = y * 30 + 15 - width * direction[0]

        # Draw the triangle
        self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, fill="#00FF00")  # Neon green

    def get_color(self, cell_type):
        # Map each cell type to a color
        colors = {
            CellType.FREE: "gray",
            CellType.WALL: "#333333",   # Dark gray
            CellType.ENEMY: "red",
            CellType.DOOR: "#8B4513",   # Saddle brown
            CellType.KEY: "yellow",
        }
        return colors.get(cell_type)