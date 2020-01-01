import numpy as np 
from class_Maze_test import Maze
from scipy.stats import binom

class NN_Model:
	def __init__(self, num_states, num_actions,hidden_neurons,learning_rate):
		mean_weight = 0
		std_weight = 2
		self.num_states = num_states
		self.num_actions = num_actions
		self.hidden_neurons = hidden_neurons
		self.learning_rate = learning_rate
		self.bias_hidden = np.random.normal(loc=mean_weight, scale=std_weight, size = (hidden_neurons,))
		self.bias_hidden = np.array(self.bias_hidden, ndmin=2).T
		self.bias_output = np.random.normal(loc=mean_weight, scale=std_weight, size = (num_actions,))
		self.bias_output = np.array(self.bias_output, ndmin=2).T
		self.W_hidden = np.random.normal(loc=mean_weight, scale=std_weight, size = (hidden_neurons,num_states))
		self.W_output = np.random.normal(loc=mean_weight, scale=std_weight, size = (num_actions,hidden_neurons))

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def predict(self, state):
		state = np.array(state.flatten(), ndmin=2).T
		output = np.dot(self.W_output,self.sigmoid(np.dot(self.W_hidden,state)+self.bias_hidden))+self.bias_output
		return(output.flatten())
		
	def train(self, state, error_term):
		state = np.array(state, ndmin=2).T
		error_term = np.array(error_term, ndmin=2).T
		output = self.predict(state)
		hidden_layer_output = self.sigmoid(np.dot(self.W_hidden,state)+self.bias_hidden)
		hidden_error = hidden_layer_output * (1 - hidden_layer_output) * np.dot(self.W_output.T,error_term)

		hidden_pd = np.dot(hidden_error,state.T)
		output_pd = np.dot(error_term,hidden_layer_output.T) 

		self.W_hidden += self.learning_rate*hidden_pd
		self.bias_hidden += self.learning_rate*hidden_error
		self.W_output += self.learning_rate*output_pd
		self.bias_output += self.learning_rate*error_term

class RL_model:
	def __init__(self,N_pos,N_actions,input_parameters,maze_type,action_selection):
		self.list_algorithms = ['QL', 'SARSA', 'AC', 'QV', 'ACLA', 'MV', 'RV', 'BM','BA']
		self.num_states = np.array([-1,1,2,2,3])*N_pos #Used for the number of inputs for the NN for maze 1 to 4 (and not 0)
		self.hidden_neurons = np.array([-1,20,60,20,100])
		self.N_actions = N_actions
		self.parameters = input_parameters
		self.action_selection = action_selection
		self.maze_type = maze_type
		self.action_selection_index = self.list_algorithms.index(action_selection)
		if (maze_type == 0):
			self.q_QL = np.zeros((N_pos,N_actions))
			self.q_SARSA = np.zeros((N_pos,N_actions))
			self.p_AC = np.zeros((N_pos,N_actions))
			self.v_AC = np.zeros(N_pos)
			self.q_QV = np.zeros((N_pos,N_actions))
			self.v_QV = np.zeros(N_pos)
			self.p_ACLA = np.ones((N_pos,N_actions))/N_actions
			self.v_ACLA = np.zeros(N_pos)
		else:
			self.q_QL = NN_Model(self.num_states[maze_type], self.N_actions,self.hidden_neurons[maze_type],self.parameters[0,0])
			self.q_SARSA = NN_Model(self.num_states[maze_type], self.N_actions,self.hidden_neurons[maze_type],self.parameters[1,0])
			self.p_AC = NN_Model(self.num_states[maze_type], self.N_actions,self.hidden_neurons[maze_type],self.parameters[2,0])
			self.v_AC = NN_Model(self.num_states[maze_type], 1,self.hidden_neurons[maze_type],self.parameters[2,1])
			self.q_QV = NN_Model(self.num_states[maze_type], self.N_actions,self.hidden_neurons[maze_type],self.parameters[3,0])
			self.v_QV = NN_Model(self.num_states[maze_type], 1,self.hidden_neurons[maze_type],self.parameters[3,1])
			self.p_ACLA = NN_Model(self.num_states[maze_type], self.N_actions,self.hidden_neurons[maze_type],self.parameters[4,0])
			self.v_ACLA = NN_Model(self.num_states[maze_type], 1,self.hidden_neurons[maze_type],self.parameters[4,1])

	def init_state(self,maze):
		maze_output = maze.get_state()
		if (self.maze_type!=1):
			return(maze_output)
		else:
			state = np.zeros(maze.maze.size)
			for i in range(len(maze.maze)):
				for j in range(len(maze.maze[0])):
					if (maze.maze[i,j] != 'W'):
						state[i*len(maze.maze[0])+j] = 1
			return(state/np.sum(state))

	def get_weights_for_boltzmann(self,state,selection_policy):
		if(selection_policy == 'QL'): 
			if (self.maze_type == 0): return(self.q_QL[state,:])
			else: return(self.q_QL.predict(state))
		elif(selection_policy == 'SARSA'): 
			if (self.maze_type == 0): return(self.q_SARSA[state,:])
			else: return(self.q_SARSA.predict(state)) 
		elif(selection_policy == 'AC'): 
			if (self.maze_type == 0):return(self.p_AC[state,:])
			else: return(self.p_AC.predict(state))
		elif(selection_policy == 'QV'): 
			if (self.maze_type == 0):return(self.q_QV[state,:])
			else: return(self.q_QV.predict(state))
		elif(selection_policy == 'ACLA'): 
			if (self.maze_type == 0): return(self.p_ACLA[state,:])
			else: return(self.p_ACLA.predict(state))

	def update_state(self,state,obs,action,maze):
		if (self.maze_type!=1):
			return(obs)
		else:
			n_cols = len(maze.maze[0])
			n_rows = len(maze.maze)
			#self.possibleActions= ["up","down", "right", "left"]
			for i in range(n_rows):
				for j in range(n_cols):
					tmp = 0
					#Contibution from up position (which is actually the up position in the matrix)
					if (maze.possibleActions[action] == 'down'):p = 0.85
					else: p = 0.05

					if (i==n_rows-1): tmp += state[i*n_cols+j]*p
					elif(maze.maze[i+1,j]=='W'): tmp += state[i*n_cols+j]*p
					else: tmp += state[(i+1)*n_cols+j]*p
						
					#Contibution from down position (which is actually the up position in the matrix)
					if (maze.possibleActions[action] == 'up'):p = 0.85
					else: p = 0.05

					if (i==0): tmp += state[i*n_cols+j]*p
					elif(maze.maze[i-1,j]=='W'): tmp += state[i*n_cols+j]*p
					else: tmp += state[(i-1)*n_cols+j]*p
							
					#Contibution from right position
					if (maze.possibleActions[action] == 'left'):p = 0.85
					else: p = 0.05
					if (j==n_cols-1): tmp += state[i*n_cols+j]*p
					elif (maze.maze[i,j+1]=='W'): tmp += state[i*n_cols+j]*p
					else: tmp += state[i*n_cols+j+1]*p

					#Contibution from left position
					if (maze.possibleActions[action] == 'right'):p = 0.85
					else: p = 0.05

					if (j==0): tmp += state[i*n_cols+j]*p
					elif (maze.maze[i,j-1]=='W'): tmp += state[i*n_cols+j]*p
					else: tmp += state[i*n_cols+j-1]*p
					
					#Calculate the number of differences in the observations and their probability
					diff_observations = 0 #Number of differences in the observations
					if(i==n_rows-1):
						if(not obs[0]): diff_observations +=1
					else:
						if(maze.maze[i+1,j]=='W' and not obs[0]): diff_observations +=1
						elif (maze.maze[i+1,j]!='W' and obs[0]): diff_observations +=1

					if(i==0):
						if(not obs[1]): diff_observations +=1
					else:
						if(maze.maze[i-1,j]=='W' and not obs[1]): diff_observations +=1
						elif (maze.maze[i-1,j]!='W' and obs[1]): diff_observations +=1

					if(j==n_cols-1):
						if(not obs[2]): diff_observations +=1
					else:
						if(maze.maze[i,j+1]=='W' and not obs[2]): diff_observations +=1
						elif (maze.maze[i,j+1]!='W' and obs[2]): diff_observations +=1

					if(j==0):
						if(not obs[3]): diff_observations +=1
					else:
						if(maze.maze[i,j-1]=='W' and not obs[3]): diff_observations +=1
						elif (maze.maze[i,j-1]!='W' and obs[3]): diff_observations +=1

					state[i*n_cols+j] = tmp+binom.pmf(diff_observations,4,0.9)
			return(state/np.sum(state))

	def update_model(self,state,action,next_state,reward,action_selection):
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
		gamma = self.parameters[0,2]
		if(self.maze_type == 0):
			alpha = self.parameters[0,0]
			self.q_QL[state,action] += alpha*(reward+gamma*np.amax(self.q_QL[next_state,:]) - self.q_QL[state,action])
		else:
			q_QL_current_state = self.q_QL.predict(state)
			q_QL_next_state = self.q_QL.predict(next_state)
			error_term = np.zeros(self.N_actions)
			error_term[action] = reward+gamma*np.amax(q_QL_next_state) - q_QL_current_state[action]
			#print(q_QL_current_state,error_term)
			self.q_QL.train(state, error_term)

	def SARSA_update(self,state,action,next_state,reward):
		alpha = self.parameters[1,0]
		gamma = self.parameters[1,2]
		prob = self.softmax_selection(self.get_weights_for_boltzmann(next_state,'SARSA'),'SARSA')
		next_action = np.random.choice(self.N_actions,p=prob)
		if(self.maze_type == 0):
			self.q_SARSA[state,action] += alpha*(reward+gamma*self.q_SARSA[next_state,next_action] - self.q_SARSA[state,action])
		else:
			q_SARSA_current_state = self.q_SARSA.predict(state)
			q_SARSA_next_state = self.q_SARSA.predict(next_state)
			error_term = np.zeros(self.N_actions)
			error_term[action] = reward+gamma*q_SARSA_next_state[next_action] - q_SARSA_current_state[action]
			self.q_SARSA.train(state, error_term)

	def AC_update(self,state,action,next_state,reward):
		alpha = self.parameters[2,0]
		beta = self.parameters[2,1]
		gamma = self.parameters[2,2]
		if (self.maze_type == 0):
			self.v_AC[state] += beta*(reward+gamma*self.v_AC[next_state]-self.v_AC[state])
			self.p_AC[state,action] += alpha*(reward+gamma*self.v_AC[next_state]-self.v_AC[state])
		else:
			v_AC_current_state = self.v_AC.predict(state)
			v_AC_next_state = self.v_AC.predict(next_state)
			error_term = reward+gamma*v_AC_next_state - v_AC_current_state
			self.v_AC.train(state, error_term)
			error_vector = np.zeros(self.N_actions)
			error_vector[action] = error_term
			self.p_AC.train(state, error_vector)			

	def QV_update(self,state,action,next_state,reward):
		alpha = self.parameters[3,0]
		beta = self.parameters[3,1]
		gamma = self.parameters[3,2]
		if (self.maze_type == 0):
			self.v_QV[state] += beta*(reward+gamma*self.v_QV[next_state]-self.v_QV[state])
			self.q_QV[state,action] += alpha*(reward+gamma*self.v_QV[next_state]-self.q_QV[state,action])
		else:
			v_QV_current_state = self.v_QV.predict(state)
			v_QV_next_state = self.v_QV.predict(next_state)
			q_QV_current_state = self.q_QV.predict(state)
			error_term = reward+gamma*v_QV_next_state - v_QV_current_state
			self.v_QV.train(state, error_term)
			error_vector = np.zeros(self.N_actions)
			error_vector[action] = reward+gamma*v_QV_next_state-q_QV_current_state[action]
			self.q_QV.train(state, error_vector)

	def ACLA_update(self,state,action,next_state,reward):
		alpha = self.parameters[4,0]
		beta = self.parameters[4,1]
		gamma = self.parameters[4,2]
		delta = reward+gamma*self.v_ACLA[next_state]-self.v_ACLA[state]
		self.v_ACLA[state] += beta*delta
		if(delta >= 0):
			for i in range(N_actions):
				if(i==action):
					self.p_ACLA[state,action] += alpha*(1-self.p_ACLA[state,action])
				else:
					self.p_ACLA[state,i] += alpha*(0-self.p_ACLA[state,i])
				if(self.p_ACLA[state,i]>1): self.p_ACLA[state,i] = 1
				elif(self.p_ACLA[state,i]<0): self.p_ACLA[state,i] = 0
		else:
			normalisation = np.sum(self.p_ACLA[state,:])-self.p_ACLA[state,action]
			for i in range(N_actions):
				if(i==action):
					self.p_ACLA[state,action] += alpha*(0-self.p_ACLA[state,action])
				else:
					if(normalisation <= 0):
						self.p_ACLA[state,i] +=  1.0/(self.N_actions-1)
					else:
						self.p_ACLA[state,i] += alpha*self.p_ACLA[state,i]*((1.0/normalisation)-1)

				if(self.p_ACLA[state,i]>1): self.p_ACLA[state,i] = 1
				elif(self.p_ACLA[state,i]<0): self.p_ACLA[state,i] = 0

	def softmax_selection (self,weight,action_selection):
		'''Returns Boltzamann distribution of preferences q_estimate and temperature t'''
		action_selection_index = self.list_algorithms.index(action_selection)
		t = 1.0/self.parameters[action_selection_index,3]
		prob = np.exp(weight/t)
		if (np.sum(np.isinf(prob))>0):
			index = np.where(np.isinf(prob))[0][0]
			prob = np.zeros(self.N_actions)
			prob[index] = 1
			return(prob)
		else:
			prob /= sum(prob)
			return(prob)

def update_reward_scores (reward,final_reward_2_addition,cum_reward_2_addition,number_episodes,interval_reward_storage,step_number):
	if ((step_number+1)%interval_reward_storage ==0 and step_number<number_episodes):
		cum_reward_2_addition += reward
	if (step_number>=number_episodes-interval_reward_storage and step_number < number_episodes):
		final_reward_2_addition += reward
	return(final_reward_2_addition,cum_reward_2_addition)

def simulation_1_epsiode(maze,action_selection,A,max_it,number_episodes,interval_reward_storage,step_number):
	if(not action_selection in A.list_algorithms): raise('action_selection is not valid')
	Total_reward = 0
	state = A.init_state(maze)
	list_position = []
	final_reward_2_addition = 0
	cum_reward_2_addition = 0
	for i in range(max_it):
		weights = A.get_weights_for_boltzmann(state,action_selection)
		prob = A.softmax_selection(weights,action_selection)
		action = np.random.choice(A.N_actions,p=prob)
		(obs,reward,won) = maze.move(action)
		list_position.append(maze.position)
		Total_reward += reward
		(final_reward_2_addition,cum_reward_2_addition) = update_reward_scores(reward,final_reward_2_addition,cum_reward_2_addition,number_episodes,interval_reward_storage,step_number)
		next_state = A.update_state(state,obs,action,maze)
		A.update_model(state,action,next_state,reward,action_selection)
		state = next_state
		if(won): break
		step_number +=1
	return(Total_reward/(i+1),list_position,A,final_reward_2_addition,cum_reward_2_addition)

def simulation_multiple_episodes(number_episodes,action_selection,max_it,N_pos,N_actions,input_parameters,interval_reward_storage,maze_type):
	# It is not clear in the paper if a learning step is defined as 1 move or 1 espisode. 
	# Therefore the final reward and cum reward are calculated for both interpretations
	# final_reward_1 and cum_reward_1 are for learning step = 1 episode
	# final_reward_2 and cum_reward_2 are for learning step = 1 move
	maze = Maze()
	A = RL_model(N_pos,N_actions,input_parameters,maze_type,action_selection)
	cum_reward_1 = 0
	final_reward_1 = 0
	cum_reward_2 = 0
	final_reward_2 = 0
	step_number = 0
	for episode in range(number_episodes):
		maze = Maze()
		if (maze_type == 0): maze.initSmallMaze()
		elif(maze_type == 1): maze.initPartObsMaze()
		elif (maze_type == 2): maze.initDynObstacMaze()
		elif(maze_type == 3): maze.initDynGoalMaze()
		elif(maze_type == 4): maze.initGenMaze()

		(average_reward,list_position,A,final_reward_2_addition,cum_reward_2_addition) = simulation_1_epsiode(maze,action_selection,A,max_it,number_episodes,interval_reward_storage,step_number)
		step_number += len(list_position)
		cum_reward_2 += cum_reward_2_addition
		final_reward_2 += final_reward_2_addition
		if((episode+1)%interval_reward_storage ==0):
			cum_reward_1 += average_reward
		if(episode>=number_episodes-interval_reward_storage):
			final_reward_1 +=average_reward
		print(step_number,average_reward)
		if(step_number>=number_episodes): break
	final_reward_1 /= interval_reward_storage
	final_reward_2 /= interval_reward_storage
	print(cum_reward_1,final_reward_1,cum_reward_2,final_reward_2)
	return(cum_reward_1,final_reward_1,cum_reward_2,final_reward_2)

maze_type = 1
N_actions = 4
N_pos = 54
max_it = 1000
maze_parameters = []
maze_parameters.append(np.array([[0.2,-1,0.9,1],[0.2,-1,0.9,1],[0.1,0.2,0.95,1],[0.2,0.2,0.9,1],[0.005,0.1,0.99,9]])) #Maze 0
maze_parameters.append(np.array([[0.02,-1,0.95,1],[0.02,-1,0.95,1],[0.02,0.03,0.95,1],[0.02,0.01,0.9,1],[0.035,0.005,0.99,10]])) #Maze 1
maze_parameters.append(np.array([[0.01,-1,0.95,1],[0.01,-1,0.95,1],[0.015,0.003,0.95,1],[0.01,0.01,0.9,0.4],[0.06,0.002,0.98,6]])) #Maze 2
maze_parameters.append(np.array([[0.005,-1,0.95,0.5],[0.008,-1,0.95,0.6],[0.006,0.008,0.95,0.6],[0.012,0.004,0.9,0.6],[0.06,0.006,0.98,10]])) #Maze 3
#maze_parameters.append(np.array([[0.01,-1,0.95,1],[0.01,-1,0.95,1],[0.015,0.003,0.95,1],[0.01,0.01,0.9,0.4],[0.06,0.002,0.98,6]])) #Maze 0

action_selection = 'QL'
number_episodes = [50000,10**5,3*(10**6),3*(10**6),15*(10**6)] #for each maze, different number of learning steps
interval_reward_storage = [2500,5000,150000,150000,750000]

simulation_multiple_episodes(number_episodes[maze_type],action_selection,max_it,N_pos,N_actions,maze_parameters[maze_type],interval_reward_storage[maze_type],maze_type)