import numpy as np 


class Maze:
    def __init__(self):
        self.possibleActions = ["up","down", "right", "left"]
    
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        # initialize empty maze
        maze=np.empty([6, 9], dtype=str)

        # Add Start
        maze[2,0]="A"

        # Add Goal
        maze[0,8]="G"

        # Add Walls
        maze[1:4,2]="W"
        maze[0:3, 7]="W"
        maze[4,5]="W"
        
        # initialize maze 
        self.maze= maze

        
    def determineAction(self, intendedAction):
        '''
        Method that determines the action.
        There is a probabilistic chance of 80 % to return the intended action.
        20 % chance to do a random action.
        '''
        
        # Generate random number between 0 and 1
        rand = np.random.random()
        
        # Check if the intended action is valid
        if intendedAction not in self.possibleActions:
            raise("ERROR. You cal only choose the actions: " + possibleActions)
        # If random number is smaller than 0.2, return random action
        elif rand<=0.2:
            return np.random.choice(self.possibleActions)
        # return intended action
        else:
            return intendedAction

    def get_position_index(self):
        position = np.where(self.maze == 'A')
        row, col = np.asscalar(position[0]), np.asscalar(position[1])
        return (row * self.maze.shape[1] + col)

    def get_position_list(self):
        position = np.where(self.maze == 'A')
        row, col = np.asscalar(position[0]), np.asscalar(position[1])
        return [row, col]

    def move(self, intended_action_index):
        '''
        Method to determine the result of a move in the maze.
        It returns next_state, reward, won
        '''
        action = self.determineAction(self.possibleActions[intended_action_index])
        position = np.where(self.maze == 'A')
        prev_row, prev_col = np.asscalar(position[0]), np.asscalar(position[1])
        row, col = prev_row, prev_col
        goal_flag = False

        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            col -= 1
        elif action == 'right':
            col += 1

        # Action takes it to a border:  -2
        if row < 0 or row >= len(self.maze) or col < 0 or col >= len(self.maze[0]):
            row, col = prev_row, prev_col
            reward = -2
        # Action takes it to wall :  -2
        elif self.maze[row, col] == 'W':
            row, col = prev_row, prev_col
            reward = -2
        # Action takes it to Goal :  100
        elif self.maze[row, col] == 'G':
            reward = 100
            self.maze[prev_row, prev_col] = ''
            # Resets the agent to the starting position
            self.maze[2, 0] = 'A'
            row, col = 2, 0
            goal_flag = True
        # Action takes it empty cell: -0.1
        else:
            reward = -0.1
            self.maze[prev_row, prev_col] = ''
            self.maze[row, col] = 'A'

        # ATTENTION!!
        # I HAVE DOUBT ABOUT THIS .......@smkj33
        state = row * len(self.maze[0]) + col

        return state, reward, goal_flag