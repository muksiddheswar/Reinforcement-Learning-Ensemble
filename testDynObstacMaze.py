from class_Maze_test import Maze
from functions import RL_model
import numpy as np

for i in range(int(1e3)):
    maze = Maze()
    maze.initDynObstacMaze()
    positionArray, wallArray = maze.get_state()
   # print("Position array: ", positionArray)
   # print()
   # print("wall array: ", wallArray)
   # print("\n\n\n")