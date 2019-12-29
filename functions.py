import numpy as np 
import matplotlib.pyplot as plt
from class_Maze_test import Maze
class RL_model:
	def __init__(self,N_states,N_actions,input_parameters):
		self.list_algorithms = ['QL', 'SARSA', 'AC', 'QV', 'ACLA', 'MV', 'RV', 'BM','BA']
		self.N_actions = N_actions
		self.parameters = input_parameters
		self.q_QL = np.zeros((N_states,N_actions))
		self.q_SARSA = np.zeros((N_states,N_actions))
		self.p_AC = np.zeros((N_states,N_actions))
		self.v_AC = np.zeros(N_states)
		self.q_QV = np.zeros((N_states,N_actions))
		self.v_QV = np.zeros(N_states)
		self.p_ACLA = np.zeros((N_states,N_actions))
		self.v_ACLA = np.zeros(N_states)

	def get_weights_for_boltzmann(self,state,selection_policy):
		if(selection_policy == 'QL'): return(self.q_QL[state,:])
		elif(selection_policy == 'SARSA'): return(self.q_SARSA[state,:])
		elif(selection_policy == 'AC'): return(self.p_AC[state,:])
		elif(selection_policy == 'QV'): return(self.q_QV[state,:])
		elif(selection_policy == 'ACLA'): return(self.p_ACLA[state,:])

	def update(self,state,action,next_state,reward,action_selection):
		if(action_selection == 'QL'): 
			self.QL_update(state,action,next_state,reward)
		elif(action_selection == 'SARSA'): 
			self.SARSA_update(state,action,next_state,reward)
		elif(action_selection == 'AC'):
			self.AC_update(state,action,next_state,reward)
		elif(action_selection == 'QV'):
			self.QV_update(state,action,next_state,reward)
		elif(action_selection == 'ACLA'):
			self.ACLA_update(state,action,next_state,reward)
		else:
			self.QL_update(state,action,next_state,reward)
			self.SARSA_update(state,action,next_state,reward)
			self.AC_update(state,action,next_state,reward)
			self.QV_update(state,action,next_state,reward)
			self.ACLA_update(state,action,next_state,reward)

	def QL_update(self,state,action,next_state,reward):
		alpha = self.parameters[0,0]
		gamma = self.parameters[0,2]
		self.q_QL[state,action] += alpha*(reward+gamma*np.amax(self.q_QL[next_state,:]) - self.q_QL[state,action])

	def SARSA_update(self,state,action,next_state,reward):
		alpha = self.parameters[1,0]
		gamma = self.parameters[1,2]
		self.q_SARSA[state,action] += alpha*(reward+gamma*self.q_SARSA[next_state,action] - self.q_SARSA[state,action])

	def AC_update(self,state,action,next_state,reward):
		alpha = self.parameters[2,0]
		beta = self.parameters[2,1]
		gamma = self.parameters[2,2]
		self.v_AC[state] += beta*(reward+gamma*A.v_AC[next_state]-A.v_AC[state])
		self.p_AC[state,action] += alpha*(reward+gamma*self.v_AC[next_state]-self.v_AC[state])

	def QC_update(self,state,action,next_state,reward):
		alpha = self.parameters[3,0]
		beta = self.parameters[3,1]
		gamma = self.parameters[3,2]
		self.v_QC[state] += beta*(reward+gamma*A.v_QC[next_state]-A.v_QC[state])
		self.q_QV[state,action] += alpha*(reward+gamma*self.v_QC[next_state]-self.q_QV[state,action])

	def ACLA_update(self,state,action,next_state,reward):
		alpha = self.parameters[4,0]
		beta = self.parameters[4,1]
		gamma = self.parameters[4,2]
		delta = reward+gamma*A.v_ACLA[next_state]-A.v_ACLA[state]
		self.v_ACLA[state] += beta*delta
		if(delta >= 0):
			for i in range(N_actions):
				if(i==action):
					self.p_ACLA[state,action] += alpha*(1-self.p_ACLA[state,action])
				else:
					self.p_ACLA[state,i] += alpha*(0-self.p_ACLA[state,i])
				if(self.p_ACLA[state,action]>1): self.p_ACLA[state,action] = 1
				elif(elf.p_ACLA[state,action]<0): self.p_ACLA[state,action] = 0
		else:
			normalisation = np.sum(self.p_ACLA[state,:])-self.p_ACLA[state,action]
			for i in range(N_actions):
				if(i==action):
					self.p_ACLA[state,action] += alpha*(0-self.p_ACLA[state,action])
				else:
					if(normalisation <= 0):
						self.p_ACLA[state,i] =  1.0/(self.N_actions-1)
					else:
						self.p_ACLA[state,i] += alpha*self.p_ACLA[state,i]((1.0/normalisation)-1)

				if(self.p_ACLA[state,action]>1): self.p_ACLA[state,action] = 1
				elif(elf.p_ACLA[state,action]<0): self.p_ACLA[state,action] = 0

	def softmax_selection (self,weight,action_selection):
		'''Returns Boltzamann distribution of preferences q_estimate and temperature t'''
		action_selection_index = self.list_algorithms.index(action_selection)
		t = 1.0/self.parameters[action_selection_index,3]
		prob = np.exp(weight/t)
		prob /= sum(prob)
		return(prob)

def simulation_1_epsiode(maze,action_selection,A,s_0,max_it = 1000):
	if(not action_selection in A.list_algorithms): raise('action_selection is not valid')
	Total_reward = 0
	state = s_0
	list_states = []
	for i in range(max_it):
		weights = A.get_weights_for_boltzmann(state,action_selection)
		prob = A.softmax_selection(weights,action_selection)
		action = np.random.choice(A.N_actions,p=prob)
		(next_state,reward,won) = maze.move(action)
		list_states.append(maze.position)
		Total_reward += reward
		A.update(state,action,next_state,reward,action_selection)
		state = next_state
		if(won): break
	return(Total_reward,list_states,A)

def simulation_multiple_episodes(number_episodes,action_selection,max_it,N_states,N_actions,input_parameters):
	maze = Maze()
	A = RL_model(N_states,N_actions,input_parameters)
	for episode in range(number_episodes):
		maze.initSmallMaze()
		s_0 = maze.get_position_index()
		(Total_reward,list_states,A) = simulation_1_epsiode(maze,action_selection,A,s_0,max_it)
		print(Total_reward)

maze_1_parameters = np.array([[0.2,-1,0.9,1],[0.2,-1,0.9,1],[0.1,0.2,0.95,1],[0.2,0.2,0.9,1],[0.005,0.1,0.99,9]])
N_actions = 4
N_states = 54
max_it = 1000
action_selection = 'QL'
number_episodes = 1000

simulation_multiple_episodes(number_episodes,action_selection,max_it,N_states,N_actions,maze_1_parameters)