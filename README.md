# DRLND-Project-Continuous-Control
Draft submission to Udacity of project code towards the completion of the nano degree "Deep Reinforcement Learning" (DRL). Project #2 called "Continuous-Control" uses reinforcement learning techniques to train an robotic arm to reach target locations.

### Introduction

The project within this repository contains code for training (an) agent(s), a double-jointed arm, to move to a target locations. 

### Project details

A reward of +0.1 is provided for each step where an agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to:
- position 
- rotation 
- velocity 
- angular velocities of the arm. 

Each **action** is a vector with four numbers, with values between -1 and 1, corresponding to torque applicable to two joints. 

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting started

Instructions for installing dependencies or downloading needed files.

1. Download the "environment" (an executable .exe file) from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacitydrlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64. zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.

- Reacher Unity environment.  
- tensorflow 1.7.1 
- numpy>=1.13.3 (instead of  1.12.1) 
- ipython 6.5.0 
- prompt-toolkit<2.0.0,>=1.0.15 (not prompt-toolkit 2.0.9)

### Instructions 

The project uses Jupyter Notebook.
To install the needed packages please see the list of requirements included in the top level directory of this repo.

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

### Agent Training Code

This repository includes functional, well-documented, and organized code for training the agents.

### License
The contents of this repository are covered under the [MIT License](LICENSE).
