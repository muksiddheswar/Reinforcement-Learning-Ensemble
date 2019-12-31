from class_Maze_test import Maze
from functions import *

maze_1_parameters = np.array([[0.2,-1,0.9,1],[0.2,-1,0.9,1],[0.1,0.2,0.95,1],[0.2,0.2,0.9,1],[0.005,0.1,0.99,9]])
N_actions = 4
N_states = 54
max_it = 1000
action_selection = 'SARSA'
number_episodes = 500
interval_reward_storage = 2500

simulation_multiple_episodes(number_episodes,action_selection,max_it,N_states,N_actions,maze_1_parameters,interval_reward_storage)