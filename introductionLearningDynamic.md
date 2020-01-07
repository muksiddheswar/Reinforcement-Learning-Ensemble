# Introduction

* What is Reinforcement learning
    - how does it work in general

Reinforcement learning is a branch of machine learning
in which software agents learn from interacting with their environment.
It is a very general framework that can be used to learn tasks of a sequential decision making nature.
An environment can exists as different states,
and the actions available to the agent depend on the state of this environment.
After every action, the agent receives a reward which might be positive or negative.
Through iterative experience, the agent seeks a policy, 
an idea about which action needs to be performed for every possible state of the environment,
that maximizes the sum of rewards over time.


    - function approximation for more complex tasks
        + Find some good source

For simple tasks,
the agent can use a tabular expression to remember values associated with certain states or state-action combinations,
also called state(-action) functions.
For complex games however, there is an explosion of state-action possibilities.
Chess for example, is estimated to have more than $10^{50}$ chess-board configurations.
Not only do we lack the memory to store such a table,
we would need more time than the age of the universe to explore all possible states.
Therefore function approximators are used to approximate such state(-action) functions.
The capture the most essential concepts in order to maximize reward.
Neural networks are an example of such function approximator.

    - some fancy real live applications
DeepMind's program AlphaZero is an interesting example that combines neural networks with reinforcement learning algorithms to learn chess.
The program was given no domain knowledge except the rules and achieved a superhuman level within 24 hours.
 
     - explain basic ones (Q-learning, SARSA, Actor-Critic)
        + Find some good sources
Some basic, well known online model-free value-function based reinforcement algorithms are SARSA and Q-learning.
They are both temporal difference (TD) reinforcement learning algorithm that learn by updating a Q-function (action value function).
The difference is that Q-learning is off-policy.
This means that the optimal action-value function is learned independent of the policy that is being followed,
unlike on-policy methods like SARSA, the learning of the action-value does depend on the policy.
The Actor-Critic (AC) is an temporal difference, on-policy learning algorithm.
But where SARSA and Q-learning only keep track of a single Q-function,
AC makes the distinction between a critic value function V that only depends on the state,
and an Actor function which will map for each action the states to preference values.
           
    - explain advanced onces with paper backing (QV-learning,ACLA)
wiering(2007) explains a more recent and advanced on-policy QV-learning method. 
This algorithm can be seen as mix between Actor-Critic and Q-learning.
As Actor-Critc, it learn the state-value function V and an action-value function.
But unlike Actor-Critic, it learns the Q-function for that.
ACLA is another on-policy RL algorithm derived from Actor-Critic
that learns a state value-function V and Actor function P.
QV-learning and ACLA have been shown to outperform similar, more basic, RL algorithms (wiering2007) and QV-learning even scores higher in certain problem contexts than more recent RL methods (wiering2009).

    - shortcomings
Reinforcement learning methods, however useful, can take many steps to learn.

<!---
shortcomings

Although the Reinforcement Learning (RL) is very used to solve problems it also has shutcomings. 
The problems we face in the real world can be extremely 
complicated in many different ways and therefore a typical Reinforcement Learning (RL) algorithm has no clue to solve. 
For example, the state space is very large in the game of Alpha GO,
environment cannot be fully observed in Poker game and there are lots of agents 
interact with each other in the real world.
--->

* Ensemble methods
<!--- 

Transition to ENSEMBLE methods

Ensemble methods are often combined with Reinforcement Learning (RL) algorithms to have a good return. 
And it is in this logic that we have chosen to work on the one paper name " Ensemble Algorithms in Reinforcement Learning".

--->

<!---

What are ensemble methods and benefits

Ensemble methods are very powerful and appropriate in the sense that when combined with Reinforcement Learning (RL) algorithms, 
they perform learning speed and final performance  when applied for solving different control problems.

--->
    - How they did it in other papers
<!---

paper on ensemble methods multi agents function approximation
						
In another paper such as  Ensemble Methods for Reinforcement Learning with Function Approximation[1], 
ensemble methods have been combined with Reinforcement Learning (RL) algorithsms.
In this paper ideas are the same like the  work that we have  elaborated with. 
The only difference that it do not have the same method as our. 
Also In this paper they describe several ensemble methods that 
combine multiple reinforcement learning algorithms for multiple agents.
For that the Temporal-Difference(TD) and  Residual-Gradient(RG) update methods 
as well as a policy function have been used . 
These two methods must be combined to the policy function 
and have been be applied to the simple pencil-and-paper game (Tic-Tac-Toe ). 
They showed that an ensemble of three agents outperforms a single agent. 
Furthermore, they performed an experiment to learn the shortest path on  a 20×20 maze.
The purpose of applying ensemble methods on games is to show that the learning speed is faster 
and from that they concluded or observed an increase in learning speed. 

--->  

<!--- 

Function approximation with neural networks

Stefan Fauber and Friedhelm Schwenker, in Neural Network Ensembles in Reinforcement Learning[3], 
propose a meta-algorithm to learn state-action values in a Neural Network Ensemble, fromed multi agent. 
The algorithm is evaluate on a generalized maze problem and on SZ-Tetris.
And  the  evaluations methods, like Temporal-Difference or SARSA, 
produce good results for problems where theYeah keep workin Markov property holds contrary 
to the methods based ona temporal-difference.

---> 
    - How we do it differently (Concept behind it)
        + find some good source material
    - Which algorithms used
<!---

Ensemble methods used in our paper

Majority Voting (VM), this one combine the best action of each algorithm and  its ﬁnal decision is based on
the number of times that an action is preferred by each algorithm. 
Rank Voting (RV), this another one lets each algorithm rank the different actions and combines these rankings to 
select a ﬁnal action.
Boltzmann mulplication(BM), uses Boltzmann exploration for each algorithm and multiplies the Boltzmann probabilities
of each action computed by each algorithm.
and Boltzmann Addition(BA), this one is similar to Boltzmann mulplication(BM) but instead to multiplie it adds the
Boltzmann probabilities of actions.

--->
 * Work they did in our paper / transition to methods   
<!---

WHAT did our paper do , transition to methods

Into the present paper we will show that several ensemble methods such as: 
Majority Voting (VM), Rank Voting, Boltzmann mulplication(BM) and Boltzmann Addition(BA) 
combine multiple different Reinforcement Learning (RL) algorithms which are: 
Q-Learning, Sarsa, Actor-Critic(AC), QV-Learning, 
and AC Learning Automaton in a single agent and the aim is to perform learning speed and final performance. 
We show  experiments on five maze problems of varying complexity.
Also one interest think to know in this paper is that in this paper
Reinforcement Learning (RL) algorithms combine are whith decision of methode instead 
to be combine by Q-value like methods in others papers.

--->









	
    
    


			
# References
			
	[1]  Stefan FauBer and Friedhelm Schwenker. Ensemble Methods for Reinforcement Learning. with Function Approximation
	[2]  Marco A. Wiering and Hado van Hasselt. The QV Family Compared to Other Reinforcement Learning Algorithms
	[3]  Stefan Fauber and Friedhelm Schwenker. Neural Network Ensembles in Reinforcement Learning
	[4]  Marco A. Wiering and Hado van Hasselt. Two Novel On-policy Reinforcement Learning Algorithms based on TD( λ )-methods
