#!/usr/bin/env python
# coding: utf-8

class Maze:
    def __init__(self):
        pass
    def initSmallMaze(self):    
        '''
        Method to generate an initial 9x6 Sutton's Dyna maze.
        '''
        pass
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
