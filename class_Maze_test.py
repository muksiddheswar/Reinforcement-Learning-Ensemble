import numpy as np 


class Maze:
    def __init__(self):
        # Not needed to initialize them to None, but it is an easy overview 
        # in one place
        self.get_state = None
        self.maze = None
        self.position = None
        self.walls=None
        self.goal = None
        self.possibleActions= ["up","down", "right", "left"] #Always true
        
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        def get_position_index():
            return(self.position[0]*self.maze.shape[1] + self.position[1])
        
        self.get_state=get_position_index
        
        self.position = [2,0]
        self.goal = [0,8]
        self.walls = [
                (1,2),
                (2,2),
                (3,2),
                (0,7),
                (1,7),
                (2,7),
                (4,5)
                ]
        
        self.generateMaze()
        
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
                        
        self.get_state = get_walls 
        
        self.position = [2,0]
        self.goal = [0,8]
        self.walls = [
                (1,2),
                (2,2),
                (3,2),
                (0,7),
                (1,7),
                (2,7),
                (4,5)
                ]
        
        self.generateMaze()
    
    def initDynObstacMaze(self):
        def get_state():
            return self.getPositionArray(), self.getWallIndexArray()
        
        self.get_state = get_state
        
        solvable_maze = False
        while not solvable_maze:
            rows = 6
            cols = 9
            indices_n = rows*cols
            obstacles_n = np.random.randint(4, high=9)
            
            self.position= [2,0]
            self.goal = [0, 8]
            
            possible_position = list(range(indices_n))
            start_index = self.coordinates2index(self.position, cols)
            goal_index = self.coordinates2index(self.goal, cols)
            for index in [start_index, goal_index]:
                possible_position.remove(index)
            wall_indices = np.random.choice(possible_position, size=obstacles_n, replace=False)
            self.walls = list()
            for wi in wall_indices:
                self.walls.append(self.index2coordinates(wi, cols))
            self.generateMaze()
            solvable_maze=self.testMaze()
    def initDynGoalMaze(self):
        def get_state():
            return self.getPositionArray(), self.getGoalArray()
        
        self.get_state = get_state
        
        self.position= [2,0]
        self.walls = [
                (1,2),
                (2,2),
                (3,2),
                (0,7),
                (1,7),
                (2,7),
                (4,5)
                ]
        
        rows = 6
        cols = 9
        position_index = self.coordinates2index(self.position, cols)
        wall_indices = list()
        for coordinate in self.walls:
            wall_indices.append(self.coordinates2index(coordinate, cols))
        goal_possible_indices = list(set(range(rows*cols)) - set([position_index]) - set(wall_indices))
        goal_index = np.random.choice(goal_possible_indices)
        self.goal = self.index2coordinates(goal_index, cols)
        self.generateMaze()
        
    
    def generateMaze(self):
        maze = np.empty([6, 9], dtype=str) 
        
        for i,j in self.walls:
            maze[i,j]="W"
        
        i = self.position[0]
        j = self.position[1]
        maze[i,j]="S"
        
        i = self.goal[0]
        j = self.goal[1]
        maze[i,j]="G"
                        
        self.maze = maze
                
    def getPositionArray(self):
        cols = len(self.maze[0])
        index = self.coordinates2index(self.position, cols)
        positionArray= [False]*self.maze.size
        positionArray[index]=True
        return positionArray
    
    def getGoalArray(self):
        cols = len(self.maze[0])
        index = self.coordinates2index(self.goal, cols)
        goalArray= [False]*self.maze.size
        goalArray[index]=True
        return goalArray
    
    def getWallIndexArray(self):
        wall_indices = list()
        for coordinate in self.walls:
            wall_indices.append(self.coordinates2index(coordinate, len(self.maze[0])))
        wallIndexArray= [False]*self.maze.size
        for wallIndex in wall_indices:
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