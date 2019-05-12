### Project criteria

The goal of this experiment is to multiple robotic arms to maintain contact with specific objects in the envionment (green spheres). The policy selected for this experiment for the agent to use is the "Actor Critic Policy Gradient Network" 
The implementation uses a UnityML based agent.  

### Code Framework 

The code is written in PyTorch and Python 3.
This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].
The specific ml-agent is called [Reacher]
(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

### Agent Goal and Actions

- Set-up: 20 Agents, common Brain, consisting of a double-jointed arm which can move to target locations.
- Goal: Each agent must move its hand to the goal location, and keep it there.
- Agent Reward Function: A reward of +0.1 is given out for each timestep when the agent's hand is in the goal location. 
- The goal of the agent is to maintain the target location for as many time steps as possible.
- The single agent slution must achieve a score of +13 averaged across all the agents for 100 consecutive episodes.
- The multiple agent solution only needs to achieve a score of 10.
- The Agent code performs an episodic task and achieve a score which exceeds 13 after 100 consecutive episodes. 
- The target number of agents for our experiment is 20.
- - Brains: Manage vector observation space and action space.

# Vector Observation Space
  - 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Visual Observations: None.
- Reset Parameters: goal size, and goal movement speed.

### Model Weights

The project includes the saved model weights of the successful agent.
The saved model weights of the successful actor are located in the file: results/results_actor_ckpt.pth'
The saved model weights of the successful critic are located in the file: results/results_critic_ckpt.pth'

### Policy-based approach

The Reacher environment uses multiple agents, performing the same task, which are all controlled by a single "brain". 
This hopefully makes the solution more efficient.

# Action space 
The action space in this experiment is "continuous" since the agent is executing fine range of movements, or action values, and not just four simple actions.  In the Udacity class there were a number policy-based methods introduced. We try and learn an optimal stochastic policy.   Policy-based methods directly learn the optimal policy, without having to storing and maintaining
all action values and the value function estimatation.   

Vector Action space: Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Learning Algorithm: Deep Deterministic Policy Gradients (DDPG) 

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

This project uses the DDPG algorithm with a replay buffer. Clearly the DQN algorithm is not sufficient to solve this problem.

DDPG:
- Agent updates the replay buffer with each experience, which is shared by all agents
- Update the local actor and critic networks using replay buffer samples
- Determine an update strategy:  
--- Every T time step ==> X times in a row (per agent) == Using S different samples from the replay buffer.
-- Uses gradient clipping when training the critic network

- Try various alternativ update strategies:  
-- Every T time step ==> Update 20 times in a rown
-- Update the networks 10 times after every 20 timesteps. 


#### Configuration of the neural network  :

The agent in our project is initialized using the hyperparameters are initialized in "model.py".
Parameters which were most influential to the agent's performance were:  TAO and BATCHSIZE

*** Please check this ****
In this project the underlying model consists of:
three fully connected layers with a rectified linear unit (ReLU) activation function on the first two layers and uses tanh after the third layer. First two layers are 400, 300 units in size.

The critic network uses 4 fully connected layers. Interestingly the critic model from the class examples used LeakyReLU instead of regular ReLU. I wasn't sure at first why the referfence model uses Leaky ReLU as the activation but my intuition is that having positive and non zero values for the value function would be adventageous.

Noise was added using an Ornstein-Uhlenbeck process (as recommended in the paper) theta and sigma were set as the same values as the paper 0.15 and 0.2 respectively.

Here is the initalization of the agent and all the hyper parameters used.

Several approaches were tried, but here are my first round settings:
* 2 hidden layers with 512 and 256 hidden units for both actor and critic
* Replay batch size 512
* Buffer size 1e6
* Replay without prioritization
* Update frequency 4
* TAU from  1e-3
* Learning rate 1e-4 for actor and 3e-4 for critic
* Ornstein-Uhlenbeck noise
* 20% droput for critic
*** Please check this ****

 
### Reward plot

A plot of rewards per episode is included to illustrate that either:

Results from the "Reacher" environment, 20 agents, goal of average above 30, 100 episodes can be found in the analytics notebook ContinuousControl.ipynb

(Option) The number of episodes needed to solve the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

Num 12th episode, with a top mean score of 39.3 in the 79th episode. The complete set of results and steps can be found in [this notebook](Continuous_Control_v8.ipynb).

<img src="results/results-graph.png" width="70%" align="top-left" alt="" title="" />
<img src="results/output.png" width="100%" align="top-left" alt="" title="Final output" />


[version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
[version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.

(Option) The number of episodes needed to solve the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

### Ideas for Future Work

Concrete future ideas for improving the agent's performance could include:
(Option) 1. A replay buffer with some kind if prioritization scheme
(Option) 2. Better exploration

## Future Improvements
There are many possible directions to try. Wish there was more time. So before anything I would just try and play with the network layers, and change different aspects of the network, units per layer, or number of layers. Not enough time left to experiment. Other algorithms I would try, which were also covered by the Udacity cource:
- I would try out the Proximal Policy Optimization algorithm.
- I would try the D3G, PPO, or D4G   algorithm (refence bellow).  

## See also
You can view the publication from DeepMind here
[ Unity Agents ] https://github.com/Unity-Technologies/ml-agents

[Distributed Distributional Deterministic Policy Gradients (D4PG)]  (https://arxiv.org/abs/1804.08617)

[ CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (section 3,7 ) https://arxiv.org/pdf/1509.02971.pdf

