from class_Maze_test import Maze
from functions import RL_model
import numpy as np

for i in range(int(1e0)):
    maze = Maze()
    maze.initDynObstacMaze()
    print(maze.get_state())
   # print("Position array: ", positionArray)
   # print()
   # print("wall array: ", wallArray)
   # print("\n\n\n")