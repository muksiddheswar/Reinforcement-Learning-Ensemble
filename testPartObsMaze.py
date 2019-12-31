from class_Maze_test import Maze
from functions import RL_model
import numpy as np

maze = Maze()
maze.initPartObsMaze()

for pos in [(0,0), (5,0), (0,8), (5,8), (0,2), (2,8), (2,5)]:
    maze.position=[*pos]
    obs = np.array([0, 0, 0, 0])
    observations = int(1e4)
    for i in range(int(observations)):
        obs += np.array(maze.get_state())
    maze.maze[pos[0],pos[1]]="P"
    print(maze.maze)
    maze.maze[pos[0],pos[1]]=""
    print("Position:", maze.position)
    print("Boundary or wall observed out of",observations)
    print("up:", obs[0])
    print("down:",obs[1])
    print("right:",obs[2])
    print("left:",obs[3])
    print("==========================================")