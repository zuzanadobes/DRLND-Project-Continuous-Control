



### Code Framework 

The code is written in PyTorch and Python 3.
This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].

### Agent Training Code

This repository includes functional, well-documented, and organized code for training the agent.


### Model Weights

The saved model weights of the successful agent are located in the file: xxxxx

### Learning Algorithm

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Agent Goal and Actions

The Agent code performs an episodic task and achieve a score which exceeds 13 after 100 consecutive episodes. The agent needs to select actions that help it to collect as many yellow bananas as possible and avoiding blue bananas.

The Agent code is capable of executing four possible actions to help him solve the task:
0 — move forward.
1 — move backward.
2 — turn left.
3 — turn right.

### Interaction with the Environment 

The agent is rewarded with +1 for collecting a yellow banana, and a reward of -1 collecting a blue banana. 

### State Space
37 dimensions which includes:
- agent’s velocity, 
- ray-based perception of objects around the agent’s forward direction. 
 
### Reward plot

The number of episodes needed to solve the environment: #
A plot of rewards per episode is included to show rewards received as the number of episodes goes over 100:

### Ideas for Future Work

Concrete future ideas for improving the agent's performance could include to perform learning directly from pixels (e.g. images). The agent not be provided pre-processed information as it is nowe.g. ray-based representation of objects around it, but would have to extract relevant information from the images directly. 

## See also

[ Unity Agents ] https://github.com/Unity-Technologies/ml-agents
