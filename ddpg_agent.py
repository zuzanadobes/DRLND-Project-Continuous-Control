# See README file for code references
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024       # minibatch size history: 1024 
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic LR_CRITIC = 1e-3    
WEIGHT_DECAY = 0.0000   # L2 weight decay (was 0.0001)
NUM_AGENTS = 20         # Better as paramter

# Added parameters to those supplied by Udacity
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
#to add next version: EPSILON_DECAY = 1e-6    # decay rate for noise process
#to add next version: LEAKINESS = 0.01
#to add next version: ACTOR_HIDDEN_SIZES = [256, 128]
#to add next version: CRITIC_HIDDEN_SIZES = [256, 256, 128]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

# Initially had n_agents as an argument.
# But it is taken care of in this version in the loop of DDPG - where we zip all the agents together, and process them one at a time 
# In the for loop. 
    def __init__(self, state_size, action_size, seed):  # Add n_agents as argument eventually
        """Initialize an Agent object.

        Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        #self.timesteps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        
        # After several attempts of playing with the timestep - I gave up. I guess it is not needed.
        # I had it as a paramter of step but removed it. I thought it would be necessary.
        #self.timestep = 0
        self.epsilon = EPSILON
        #self.n_agents = NUM_AGENTS
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # For n_agents 
    # def step(self, state, action, reward, next_state, done, timestep=1)
    # Not used - will see if it can be used in future version
    def step_multi(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # for one agent:
        # #self.memory.add(state, action, reward, next_state, done)
        # For multiple agents:
        self.timesteps += 1
        for i in range(self.n_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])        
        
        # Learn at defined interval, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
 
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        ##self.memory.add(state, action, reward, next_state, done)
        #self.timesteps += 1
        self.memory.add(state, action, reward, next_state, done)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()    # try to make this work action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    # Not used - will see if it can be used in future version
    def act_multi(self, states, add_noise=True):
        """ Given a list of states for each agent it returns [n_agents, n_actions]
            representing actions taken by each agent based on the current policy.
            NOTE: clips actions to be between -1, 1
        Args:
            states:    () one row of state for each agent 
            add_noise: (bool) add noise to the actions?
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += [self.noise.sample() for _ in range(self.n_agents)]
        return np.clip(actions, -1, 1)


    def reset(self):
        self.noise.reset()
        
    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Several steps now need to be done for the critic to complete the learning:
        # - from target models get the predicted next-state actions and Q values 
        # - calculate the Q targets for current states (y_i)
        # - determine loss and minimize it
        # - minimize the loss
        # - update target networks
        # - update noise 
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # optimization of loss:
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # ? Not exactly sure what this does.
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Next in line is the actor:
        # - update actor and determine loss
        # - minimize the loss
        # - update target networks
        # - update noise 
        # - update the network weights 
        # - reset the loss 
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update the weights
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        ##self.epsilon -= EPSILON_DECAY # ==> next iteration work with epsilon 
        #?self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu: long-running mean
            theta: the speed of mean reversion
            sigma: the volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
