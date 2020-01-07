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
This could be one of the reason why our neural networks were not able to converge,
while in the original paper, they did.
Other possible explanation are numerical errors. 

* Small maze
    - Very similar results
        + within standard deviation

(Wiering 2008) determined all learning parameters by performing various trials with different parameter values.
They noted in their paper that the discount factor had a major effect on the performance.
To assess this effect,
the SARSA and the ACLA algorithms were trained on the simple maze with changed discount.
We observe a significant change in the final and the cumulative reward,
which highlights the importance of this parameter in the overall performance of the algorithms. 


For the Dynamic Maze, the results  were inconclusive. The reason behind this can be attrbuted the
