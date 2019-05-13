### Project criteria

The goal of this experiment is to multiple robotic arms to maintain contact with specific objects in the envionment (green spheres). The policy selected for this experiment for the agent to use is the "Actor Critic Policy Gradient Network". 
The implementation uses a UnityML based agent.  The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  After each episode, we add up the rewards that each agent received, to get their average scores, and then take the average of these.  This finally is used to compute the episode average score.  The environment is considered solved, when the average  goes over 30,  after 100 episodes.

### Code Framework 

The code is written in PyTorch and Python 3.
This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].
The specific environment for this project came in the form of a windows executable file,  is called [Reacher]
(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

# Vector Observation Space
  - 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Visual Observations: None.
- Reset Parameters: goal size, and goal movement speed.

### Model Weights

The project includes the saved model weights of the successful agent.
The saved model weights of the successful actor are located in the file: results/results_actor_ckpt.pth'
The saved model weights of the successful critic are located in the file: results/results_critic_ckpt.pth'

### Policy-based approach

The Reacher environment uses multiple agents, performing the same task, which are all controlled by a single "brain".  The DDPG algorrithm that is used in this project,  collects information from the observation space, using a method called the policy gradient method. For problems with continuous action space as is the case in our problem, a policy oriented approach is used more than value-based approaches. To help make the policy obtaining process more effective a Actor-Critic, where the Critic helps the Actor with performance related information, while the Actor learns.


### Agents

- Agents correspond to a double-jointed arm which can move to target locations.
- Goal: Each agent must move its hand to the goal location, and keep it there.
- Agent Reward Function: A reward of +0.1 is given out for each timestep when the agent's hand is in the goal location. 
- The goal of the agent is to maintain the target location for as many time steps as possible.
- The single agent slution must achieve a score of +13 averaged across all the agents for 100 consecutive episodes.
- The multiple agent solution needs to achieve a moving average score of 30 over all agents, over all episodes.

### Action space 

The action space in this experiment is "continuous" since the agent is executing fine range of movements, or action values, and not just four simple actions.  In the Udacity class there were a number policy-based methods introduced. We try and learn an optimal stochastic policy.   Policy-based methods directly learn the optimal policy, without having to storing and maintaining
all action values and the value function estimatation.   

Vector Action space: Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Learning Algorithm: Deep Deterministic Policy Gradients (DDPG) 

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

This project uses the DDPG algorithm with a replay buffer. Clearly the DQN algorithm is not sufficient to solve this problem.
The agent is initialized using the hyperparameters are initialized in "ddpg_agent.py".
Parameters which were most influential to the agent's performance were:  BATCHSIZE but in general it is very time consuming to measure the effects. TAO setting for the neural net also effected the outcome.


- Set-up: 20 Agents, using one common Brain. Brains manage vector observation space and action space.
- Agent updates the replay buffer with each experience, which is shared by all agents
- Update the local actor and critic networks using replay buffer "samples"
- the update strategy:  
-- Every T time step ==> X times in a row (per agent) == Using S different samples from the replay buffer.
--- Uses gradient clipping when training the critic network
--- I tried various alternativ update strategies - and will be more precise in the second version of this code as to the differences. In this version it was pure survival coding.
---- Every T time steps update the network 20 times in a row
---- Update the networks 10 times after every 20 timesteps. 

### The neural network 

The network comprises of 2 networks and the settings are described in the model.py file. 
The network architecture used by the Actor and Critic consist of three fully connected layers, with 400 units and 300 units. 
The Actor uses the ReLU activation functions & Critic uses the LeakyReLU activation function and they use tanh on the output layer. 

Noise was added using an Ornstein-Uhlenbeck process theta and sigma were set as the recommended values from classroom reading. (See Also)

Some of the hyper parameters used, several approaches were tried:
* Replay batch size 512
* Buffer size 1e6
* Replay without prioritization
* TAU from  1e-3
* Learning rate 1e-4 for actor and 3e-4 for critic
* Ornstein-Uhlenbeck noise
 
### Reward plot

A plot of rewards per episode is included to illustrate that either:

Results from the "Reacher" environment, 20 agents, goal of average above 30, 100 episodes can be found in the analytics notebook ContinuousControl.ipynb

(Option) The number of episodes needed to solve the environment: ###
(Option) The top mean score achieved while solving the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

### Ideas for Future Work

Concrete future ideas for improving the agent's performance could include:
(Option) 1. A replay buffer with some kind if prioritization scheme
(Option) 2. Better exploration

## Future Improvements
There are many possible directions to try. Wish there was more time. So before anything I would just try and play with the network layers, and change different aspects of the network, units per layer, or number of layers. Not enough time left to experiment. Other algorithms I would try, which were also covered by the Udacity cource:
- I would try out the Proximal Policy Optimization algorithm.
- I would try the D4G algorithm (refence bellow).  

## See also
You can view the publication from DeepMind here
[ Unity Agents ] https://github.com/Unity-Technologies/ml-agents

[Distributed Distributional Deterministic Policy Gradients (D4PG)]  (https://arxiv.org/abs/1804.08617)

[ CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (section 3,7 ) https://arxiv.org/pdf/1509.02971.pdf

[ Henry Chan's git Repository ] https://github.com/kinwo/deeprl-continuous-control
