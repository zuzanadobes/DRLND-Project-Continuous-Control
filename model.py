# Adopted from https://github.com/tommytracey/DeepRL-P2-Continuous-Control/blob/master/model.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Tried also: fc1_units=288, fc2_units=288
# fc1_units (int): Number of nodes in first hidden layer
# fc2_units (int): Number of nodes in second hidden layer
# self.fc1 = nn.Linear(state_size, fc1_units) 
# self.bn1 = nn.BatchNorm1d(fc1_units)

# self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.fc3 = nn.Linear(fc2_units, action_size)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed = 2, fc1_units=400, fc2_units=300):    # Try also simpler without the fc2_units
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        #self.fc1 = nn.Linear(state_size, fc_units) # fc1
        #self.fc2 = nn.Linear(fc_units, action_size)
        #self.reset_weights()
        self.fc1 = nn.Linear(state_size, fc1_units)
        #
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_weights()
        
#When agent uses second,third layer
#self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

# Alternative:
# x = F.relu(self.bn1(self.fc1(state)))
# x = F.relu(self.fc2(x))
# return F.tanh(self.fc3(x))
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return tanh(self.fc3(x))

# Critic as neural network
# Critic: 256 -> 256 -> 128
# Alternatively tried : fc1_units=400, fc2_units=300,no fc3_units
#self.bn1 = nn.BatchNorm1d(fc1_units)
# self.fc3 = nn.Linear(fc2_units, 1)
class Critic(nn.Module):
    """Critic (Value) Model."""

    
    #def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256,fc3_units=128):
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_weights()

    def reset_weights(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
