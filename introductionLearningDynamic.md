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
Ensemble methods are a powerful method to combine different Reinforcement Learning (RL) algorithms, 
which often result in improved learning speed and final performance.

    - How they did it in other papers
[1] used ensemble methods to combine multiple reinforcement learning algorithms for multiple agents for which they used Temporal-Difference(TD) and  Residual-Gradient(RG) update methods as well as a policy function.
Other ensemble methods have been used in reinforcement learning to combine value functions stored by function approximators (wiering 2008 [14], [15], [16], [17]).
However, only Rl algorithms with the same value function can be combined in this way.
(Wiering2008) wanted to combine RL methods with different value functions and policies (e.g. Q-learning and ACLA).
It is possible however, 
to combine the different policies that were derived from distinct value functions.
Some algorithms that perform this task and take exploration into account at the same time are Majority voting, Rank voting, Boltzmann multiplication, and Boltzmann addition.
Majority Voting (VM) combines the best action of the RL algorithms and bases the final decision on the number of times that each action was preferred by the different RL methods.
Rank Voting (RV) lets each algorithm rank the different actions and combines these rankings into ﬁnal preferences over actions.
Boltzmann mulplication(BM) multiplies the Boltzmann probabilities
of each action computed by each RL algorithm.
Finally, Boltzmann Addition(BA) is very similar to Boltzmann mulplication, but adds instead of multiplies the Boltzmann probabilities of actions.

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
