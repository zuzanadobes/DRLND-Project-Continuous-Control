# DRLND-Project-Continuous-Control
Draft submission to Udacity of project code towards the completion of the nano degree "Deep Reinforcement Learning" (DRL). Project #2 called "Continuous-Control" uses reinforcement learning techniques to train an robotic arm to reach target locations.

### Introduction

For this project, you will train an agent, a double-jointed arm, move to target locations. 

### Project details

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting started

Instructions for installing dependencies or downloading needed files.
This project uses Reacher Unity environment.  The following modules are needed:<br>
tensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible.
ipython 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 2.0.9 which is incompatible.

### Instructions 

A description of how to run the code in the repository, to train the agent.

The project uses Jupyter Notebook.
To install the needed packages please see the list of requirements included in the top level directory of this repo.

## Not true currently 
The project uses Jupyter Notebook. This command needs to be run to install the needed packages:



Instructions
The project consists of 8 files:

### Repository structure

The code is structured as follows:

ContinuousControlProject.ipynb - run this file in Jupyter Notebook
agent.py - the DDPG Agent class
network.py - the Actor and Critic models
memory.py - the replay buffer class
noise.py - the noise class
config.py - the configuration class for configuring the project (hyperparameters, network, optimizers etc.)
checkpoint_critic.pth - critic trained model
checkpoint_actor.pth - actor trained model
Report.md - description of the implementation

Use Config() class is 

### License
The contents of this repository are covered under the [MIT License](LICENSE).
