from class_Maze_test import Maze
from functions import RL_model
import numpy as np

for i in range(int(1e0)):
    maze = Maze()
    maze.initGenMaze()
    print(maze.get_state())
    #print(maze.maze)
    #print("\n\n\n")