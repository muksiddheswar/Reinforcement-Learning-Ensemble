import numpy as np 


class Maze:
    def __init__(self):
        # Not needed to initialize them to None, but it is an easy overview 
        # in one place
        self.get_state = None
        self.maze = None
        self.position = None
        self.wall_indices=None
        self.possibleActions= ["up","down", "right", "left"] #Always true
        
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        def get_position_index():
            return(self.position[0]*self.maze.shape[1] + self.position[1])
        
        # Assign self.get_state function
        self.get_state = get_position_index 
        
        # initialize empty maze
        maze=np.empty([6, 9], dtype=str) 

        # Add start
        self.position = [2,0]
        maze[self.position[0],self.position[1]]="S"
        
        # Add Goal
        maze[0,8]="G" 

        # Add Walls
        maze[1:4,2]="W"
        maze[0:3, 7]="W"
        maze[4,5]="W"
        
        # initialize maze 
        self.maze= maze
        
    def initPartObsMaze(self):
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        def get_walls():
            '''
            Return a list of integeres for up, down, right, left
            1 means there is a wall or boundary
            0 means there is nothing
            '''
            up = np.copy(self.position)
            up[0]+=1            
            down = np.copy(self.position)
            down[0]-=1
            right = np.copy(self.position)
            right[1]+=1
            left = np.copy(self.position)
            left[1]-=1
            possibleActionPositions=[up, down, right, left]
            
            state=list()
            for actionPosition in possibleActionPositions:
                rand = np.random.random()
                if rand <= 0.10:
                    # return random if there is a wall or boundary or not
                    state.append(np.random.choice([0,1]))
                else:
                    i = actionPosition[0]
                    j = actionPosition[1]
                    rows = len(self.maze)
                    cols = len(self.maze[0])
                    # Boundary condition
                    if i < 0 or j < 0 or i >= rows or j >= cols:
                        state.append(1)
                    # Wall condition
                    elif self.maze[i,j] == "W":
                        state.append(1)
                    else:
                        state.append(0)
            return state
                        
            
        
        # Assign self.get_state function
        self.get_state = get_walls 
        
        # initialize empty maze
        maze=np.empty([6, 9], dtype=str) 
        
        # Add start
        self.position = [2,0]
        maze[self.position[0],self.position[1]]="S"
        
        # Add Goal
        maze[0,8]="G" 

        # Add Walls
        maze[1:4,2]="W"
        maze[0:3, 7]="W"
        maze[4,5]="W"
        
        # initialize maze 
        self.maze= maze
    
    def initDynObstacMaze(self):
        def get_state():
            return self.getPositionArray(), self.getWallIndexArray()
        self.get_state = get_state
        solvable_maze = False
        while not solvable_maze:
            rows = 6
            cols = 9
            maze=np.empty([rows, cols], dtype=str)
            indices_n = rows*cols
            obstacles_n = np.random.randint(4, high=9)
            possible_position = list(range(indices_n))
            
            # Add start to maze
            self.position= [2,0]
            start_index = self.coordinates2index(self.position, cols)
            maze[self.position[0],self.position[1]]="S"
            
            
            # Add Goal to maze
            goal_coordinates = [0, 8]
            goal_index = self.coordinates2index(goal_coordinates, cols)
            maze[goal_coordinates[0],goal_coordinates[1]]="G"
            
            # Add walls to maze
            for index in [start_index, goal_index]:
                possible_position.remove(index)
            self.wall_indices = np.random.choice(possible_position, size=obstacles_n, replace=False)
            for wi in self.wall_indices:
                coordinates = self.index2coordinates(wi, cols)
                i = coordinates[0]
                j = coordinates[1]
                maze[i,j]="W"
            self.maze= maze
            solvable_maze=self.testMaze()
            
    def getPositionArray(self):
        cols = len(self.maze[0])
        index = self.coordinates2index(self.position, cols)
        positionArray= [False]*self.maze.size
        positionArray[index]=True
        return positionArray
    def getWallIndexArray(self):
        wallIndexArray= [False]*self.maze.size
        for wallIndex in self.wall_indices:
            wallIndexArray[wallIndex]=True
        return wallIndexArray
        
    def testMaze(self):
        '''
        Test whether the maze is solvable.
        Walls may not obstruct the path from the current position of the agent to the goal
        '''
        rows= len(self.maze)
        cols= len(self.maze[0])
        states2check = {self.coordinates2index(self.position, cols)}
        states_checked = set()
        while len(states2check)>0:
            #print("states to check:", states2check)
            #print("states checked:", states_checked)
            #print()
            state = states2check.pop()
            states_checked.add(state)
            i,j = self.index2coordinates(state, cols)
            if self.maze[i,j]=="G":
                return True
            elif self.maze[i,j]!="W":
                states2check = (states2check | set(self.getNeighbourIndices(state,rows,cols))) - states_checked
        return False
    
    def index2coordinates(self, index, cols):
        return (index // cols, index % cols)
    
    def coordinates2index(self, coordinates, cols):
        i = coordinates[0]
        j = coordinates[1]
        index = i*cols + j
        return index
    
    def getNeighbourCoordinates(self, coordinates, rows, cols):
        i = coordinates[0]
        j = coordinates[1]
        neighbours = list()
        for newCoordinates in [(i,j+1),(i,j-1),(i+1,j),(i-1,j)]:
            iNeighbour=newCoordinates[0]
            jNeighbour=newCoordinates[1]
            if not (iNeighbour < 0 or jNeighbour < 0 or iNeighbour >= rows or jNeighbour >= cols):
                neighbours.append(newCoordinates)
        return neighbours
                
    
    def getNeighbourIndices(self, index, rows, cols):
        coordinates = self.index2coordinates(index, cols)
        neighbours = self.getNeighbourCoordinates(coordinates, rows, cols)
        neighbour_indices=list()
        for neighbour_coordinates in neighbours:
            neighbour_indices.append(self.coordinates2index(neighbour_coordinates, cols))
        return neighbour_indices
            
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