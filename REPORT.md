### Project criteria

To train an agent to carry out specific kinds of movements.
The policy selected for this agent to use is the "Actor Critic Policy Gradient Network" 
The implementation uses a UnityML based agent.

### Code Framework 

The code is written in PyTorch and Python 3.
This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].

### Agent Training Code

This repository includes functional, well-documented, and organized code for training the agent.


### Model Weights

The project includes the saved model weights of the successful agent.
The saved model weights of the successful agent are located in the file: xxxxx

### Learning Algorithm: Deep Deterministic Policy Gradients (DDPG) 

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

This project uses the DDPG algorithm with a replay buffer. 

Version 1: (Udacity)
DDPG for multiple agents.  With each step:
- Agent updates the replay buffer with experience, sharable by all agents
- Update the local actor and critic networks using replay buffer samples
-- Determine an update strategy:  
--- Every T time step ==> X times in a row (per agent) == Using Y different samples from the replay buffer.
--- e.g. 20 times at every timestep, we amended the code to update the networks 10 times after every 20 timesteps. 
-- Use gradient clipping when training the critic network:
 * self.critic_optimizer.zero_grad()
 * critic_loss.backward()
 * torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
 * self.critic_optimizer.step()

#### Configuration deployed: (sample)
* 2 hidden layers with 512 and 256 hidden units for both actor and critic
* Replay batch size 512
* Buffer size 1e6
* Replay without prioritization
* Update frequency 4
* TAU from  1e-3
* Learning rate 1e-4 for actor and 3e-4 for critic
* Ornstein-Uhlenbeck noise
* 20% droput for critic

### Agent Goal and Actions

The Agent code performs an episodic task and achieve a score which exceeds 13 after 100 consecutive episodes. The agent needs to select actions that help it to collect as many yellow bananas as possible and avoiding blue bananas.

The Agent code is capable of executing four possible actions to help him solve the task:

### Interaction with the Environment 

The agent is rewarded with +1 for collecting a yellow banana, and a reward of -1 collecting a blue banana. 

### State Space
37 dimensions which includes:
- agent’s velocity, 
- ray-based perception of objects around the agent’s forward direction. 
 
### Reward plot

A plot of rewards per episode is included to illustrate that either:

[version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
[version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.

(Option) The number of episodes needed to solve the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

### Ideas for Future Work

Concrete future ideas for improving the agent's performance could include:
(Option) 1. A replay buffer with some kind if prioritization scheme
(Option) 2. Better exploration

## See also
You can view the publication from DeepMind here
[ Unity Agents ] https://github.com/Unity-Technologies/ml-agents

[ CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING (section 3,7 ) https://arxiv.org/pdf/1509.02971.pdf
