from class_Maze_test import Maze
from functions import RL_model
import numpy as np

maze = Maze()

for i in range(100):
    maze.initDynObstacMaze()
    print(maze.maze)
    print()