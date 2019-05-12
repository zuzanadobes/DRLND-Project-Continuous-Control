# DRLND-Project-Continuous-Control
Draft submission to Udacity of project code towards the completion of the nano degree "Deep Reinforcement Learning" (DRL). Project #2 called "Continuous-Control" uses reinforcement learning techniques to train an robotic arm to reach target locations.

### Introduction

The project within this repository contains code for training (an) agent(s), a double-jointed arm, to move to a target locations. 

Multiple agents are used, and 20 deployed agents must get an average score of +30 over all agents, over 100 consecutive episodes.
After each episode, the scores are tabulated for each agent by adding up the rewards of each agent without any discounting. 
The ** average score ** of all agents for each espisode is taken.

The environment is considered "solved", when the average (over 100 episodes) of those average scores is at least **+30**.

### Project details

A reward of +0.1 is provided for each step where an agent's arm is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to:
- position 
- rotation 
- velocity 
- angular velocities of the arm. 

Each **action** is a vector with four numbers, with values between -1 and 1, corresponding to torque applicable to two joints. 

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting started

Instructions for installing dependencies or downloading needed files.

1. Download the Reacher Unity "environment" (an executable .exe file) from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacitydrlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in your (DRLND) GitHub repository, in your working directory of your project e.g. `p2_continuous-control/` folder, and unzip (or decompress) the file.
3. All required modules for this project are listed in requirements.txt. You can load them into your project environment with the command:  conda create -n your_environment --file requirements.txt
Mo
- tensorflow 1.7.1 
- numpy=1.16.3=pypi_0  # or numpy>=1.13.3 (instead of  1.12.1) 
- ipython=7.5.0=pypi_0  # or ipython 6.5.0 
- prompt-toolkit=2.0.9=pypi_0 # or prompt-toolkit<2.0.0,>=1.0.15 (not prompt-toolkit 2.0.9) 

### Instructions 

The project uses Jupyter Notebook as the central walkthrough code part.
To install the needed packages please see the list of requirements included in the top level directory of this repo.

### Repository structure

The code is structured as follows:

Continuous_Control.ipynb - run this file in Jupyter Notebook
ddpg_agent.py - the DDPG Agent class and the OUNoise class and the ReplayBuffer class
model.py - the Actor and Critic models
results_actor_ckpt.pth - actor trained model
critic.pth - critic trained model
report.md - description of the first draft of the implementation.
The configuration (hyperparameters, network, optimizers etc.) for configuring the project are located in the report.md file.

Use Config() class is 

### Agent plus Training Code

This repository includes functional, well-documented, and organized code for training the agents.

The ddpg_agent.py contains code for this, which is based on the version supplied by udacity:
https://github.com/udacity/deep-reinforcement-learning/blob/55474449a112fa72323f484c4b7a498c8dc84be1/ddpg-bipedal/ddpg_agent.py 

Some code parts inspired by the following code repositories: 
https://github.com/tommytracey/DeepRL-P2-Continuous-Control
https://github.com/RicardoCRubio/Udacity_DRLND_Continuous_Control


### License
The contents of this repository are covered under the [MIT License](LICENSE).
