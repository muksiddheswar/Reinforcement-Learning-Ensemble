#!/usr/bin/env python
# coding: utf-8

import numpy as np

class Maze:
    def __init__(self):
        pass
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        # initialize empty maze
        maze=np.empty([6, 9], dtype=str)

        # Add Start
        maze[2,0]="S"

        # Add Goal
        maze[0,8]="G"

        # Add Walls
        maze[1:4,2]="W"
        maze[0:3, 7]="W"
        maze[4,5]="W"
        
        # initialize maze 
        self.maze= maze
        
        
    def initPartObservMaze(self):
        '''
        Method to generate an initial 9x6 partially observable maze.
        '''
        pass
    def initDynObstaclMaze(self):
        '''
        Method to generate an initial 9x6 maze with dynamically allocated obstacles.
        '''
        pass
    def initDynGoalMaze(self):
        '''
        Method to generate an initial 9x6 maze with a dynamically allocated goal.
        '''
        pass
    def initGeneralMaze(self):
        '''
        Method to generate an initial 9x6 maze with dynamically allocated obstacles and a dynamically allocated goal.
        '''
        pass
    def generateRewards(self, state, action):
        '''
        Method that determines reward based on the state and action.
        '''
        pass
    def generateNextState(self, state, action):
        '''
        Method that determines how a current state will transition into a new state, based on a given action.
        '''
        pass
    def determineAction(self, intendedAction):
        '''
        Method that determines the action.
        There is a probabilistic chance of 80 % to return the intended action.
        20 % chance to do a random action.
        '''
        pass
    def move(self, state, action):
        '''
        Method to determine the result of a move in the maze.
        It returns next_state, reward, won
        '''
        # (next_state,reward,won) = maze.move(state,action)