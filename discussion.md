# Discussion

Although the article of (wiering 2008) is very interesting and shows new insights,
we are of the opinion that they could have taken extra steps to assure reproducibility.
Some methods were ambiguously explained, so that it took some effort to figure out what was meant exactly.
The source was also not available, which would have solved previous remark.
For example,
while implementing the neural network,
it is not clear how the weights are updates.
We decided to update the weights after each move gradient descent for one iteration.
However, it is possible that in the original paper,
they used multiple gradient descent iterations after each move.
Or maybe, action replay as in, "Human-level control through deep reinforcement learning",
from DeepMind where used. 
Here, actions, states and rewards are stored in memory and every so often, the weights are updated by learning from the experiences stored in memory.


* There was still some ambiguity in the article that could have been explained better
    - source code was not available (or we did not find it at least)
    - stuff they did not explain about the neural network that was needed for the implementation
        + I don't know exactly what stuff, can you help me out Nil?
    - not totally clear how final and cumulative score are implemented
    
    - Maybe at the end of all this negative notes, tell that it was still very interesting work and that we managed to find most things after searching for a while

* Explain that they use cumulative results and final results
    - That cumulative gives an idea about how fast the algo learns but graph would be better
    
* why neural nets do not converge
    - numerical errors
    - normal multiple iteration before the forward and backward propagation
    - something with experience replay
    

* Small maze
    - Very similar results
        + within standard deviation
    - we can tell something about the rank voting that there is still a minor bug that could have influence
    - variation of parameters
        + gamma 






(Wiering 2008) determined all learning parameters by performing various trials with different parameter values.
They noted in their paper that the discount factor had a major effect on the performance.
To assess this effect,
the SARSA and the ACLA algorithms were trained on the simple maze with changed discount.
We observe a significant change in the final and the cumulative reward,
which highlights the importance of this parameter in the overall performance of the algorithms. 





Due to lack of computational power and time, this observation was generated for 1 simulation only. As a result small differences in performance of the algorithms were not clearly distinguishable. Hence it was not possible to compare the convergence of the algorithms other than identify the slowest.





For the Dynamic Maze, the results  were inconclusive. The reason behind this can be attrbuted the
