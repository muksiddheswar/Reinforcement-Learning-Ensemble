from class_Maze_test import Maze
from functions import RL_model
import numpy as np

for i in range(int(1e1)):
    maze = Maze()
    maze.initDynGoalMaze()
    positionArray, goalArray = maze.get_state()
    print("Position array: ", positionArray)
    print()
    print("Goal array: ", goalArray)
    print("\n\n\n")