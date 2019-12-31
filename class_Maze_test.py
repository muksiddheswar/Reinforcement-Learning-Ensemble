import numpy as np 


class Maze:
    def __init__(self):
        self.get_state=None
    
    
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        def get_position_index():
            return(self.position[0]*self.maze.shape[1] + self.position[1])
        
        self.get_state = get_position_index
        # initialize empty maze
        maze=np.empty([6, 9], dtype=str)

        # Add Start
        #maze[2,0]="A" # Not needed I think

        # Add Goal
        maze[0,8]="G"

        # Add Walls
        maze[1:4,2]="W"
        maze[0:3, 7]="W"
        maze[4,5]="W"
        
        # initialize maze 
        self.maze= maze
        self.position = [2,0]
        self.possibleActions= ["up","down", "right", "left"]
        
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

    def move(self, intended_action_index):
        '''
        Method to determine the result of a move in the maze.
        It returns next_state, reward, won
        '''
        action = self.determineAction(self.possibleActions[intended_action_index])
        next_position = np.copy(self.position)
        if(action == 'up'):
            next_position[0] +=1
        elif(action == 'down'):
            next_position[0] -=1
        elif(action == 'left'):
            next_position[1] -=1
        else:
            next_position[1] +=1

        if(next_position[0]<0 or next_position[0]>=self.maze.shape[0] or next_position[1]<0 or next_position[1]>=self.maze.shape[1]):
            return(self.get_state(),-2,False)
        elif(self.maze[next_position[0],next_position[1]] == 'W'):
            return(self.get_state(),-2,False)
        elif(self.maze[next_position[0],next_position[1]] == 'G'):
            self.position = next_position
            return(self.get_state(),100,True)
        else:
            self.position = next_position
            return(self.get_state(),-0.1,False)