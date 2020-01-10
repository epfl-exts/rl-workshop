from collections import defaultdict, Counter
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import random
import torch
import gym.spaces as spaces

def set_seed(env, seed):
    """Helper function to set the seeds when needed"""
    env.seed(seed) # Environment seed
    env.action_space.seed(seed) # Seed for env.action_space.sample()
    np.random.seed(seed) # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed
    random.seed(seed) # seed for Python random library

class MultiAgentTrainer():
    """A class to train agents in a multi-agent environment"""
    def __init__(self, env, agents, seed=None):
        # Save parameters
        self.env, self.agents, self.seed = env, agents, seed
        
        # Create log of rewards and reset agents
        self.rewards_log = {name: [] for name in self.agents.keys()}
        self.reset()
        
    def reset(self):
        # Set seed for reproducibility
        if self.seed is not None:
            set_seed(self.env, self.seed)
        
        # Reset agents and clear log of rewards
        for name, agent in self.agents.items():
            agent.reset()
            self.rewards_log[name].clear()

    def train(self, n_steps):
        # Reset env. and get initial observations
        states = self.env.reset()
        
        for i in tqdm_notebook(range(n_steps), 'Training agents'):
            # Select actions based on current states
            actions = {name: agent.act(states[name]) for name, agent in self.agents.items()}

            # Perform the selected action
            next_states, rewards, dones, _ = self.env.step(actions)

            # Learn from experience
            for name, agent in self.agents.items():
                agent.learn(states[name], actions[name], rewards[name], next_states[name], dones[name])
                self.rewards_log[name].append(rewards[name])
            states = next_states
    
def test_agents(env, agents, n_steps, seed=None):
    """Function to test agents"""
    
    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()
    rewards_log = defaultdict(list)
    
    for _ in tqdm_notebook(range(n_steps), 'Testing agents'):
        # Select actions based on current states
        actions = {name: agent.act(states[name]) for name, agent in agents.items()}
        
        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)
        
        # Save rewards
        for name, reward in rewards.items():
            rewards_log[name].append(reward)
        
        states = next_states

    return rewards_log

def plot_cumulative_rewards(rewards_log, ax=None, subset=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
        
    # Define which entry to plot
    subset = rewards_log.keys() if subset is None else subset
    
    # Plot rewards
    for name, rewards in rewards_log.items():
        if name in subset:
            # Work with Numpy array
            rewards = np.array(rewards)
            pickup = (rewards == 1)
            crashed = (rewards == -1)

            # Compute cumulative sum
            cumsum = np.cumsum(rewards)
            idxs = range(1, len(cumsum) + 1)

            # Create label with pickup/crash rate
            label = r'{} - reward: {:.3f}±{:.3f}, pickup: {:.2f}% ({}) crash: {:.2f}% ({})'.format(
                name, np.mean(rewards), np.std(rewards),
                100*pickup.mean(), pickup.sum(), 100*crashed.mean(), crashed.sum())

            # Plot results
            ax.step(idxs, cumsum, label=label)

    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative reward')
    ax.legend()
    
    if create_figure:
        plt.show()
    
def plot_rolling_rewards(rewards_log, ax=None, window=None, hline=None, subset=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
        
    # Define which entry to plot
    subset = rewards_log.keys() if subset is None else subset
        
    for name, rewards in rewards_log.items():
        if name in subset:
            # Work with Numpy array
            rewards = np.array(rewards)
            steps = range(1, len(rewards)+1)
            pickup = (rewards == 1)
            crashed = (rewards == -1)

            # Set default for window size
            window = int(len(rewards)/10) if window is None else window

            # Plot rolling mean
            rolling_mean = pd.Series(rewards).rolling(window).mean()
            label = '{} - pickup: {} crash: {}'.format(name, pickup.sum(), crashed.sum())
            ax.plot(steps, rolling_mean,label=label)
        
    if hline is not None:
        ax.axhline(hline, label='target value', c='C0', linestyle='--')

    # Add title, labels and legend
    ax.set_xlabel('Steps (rolling window: {})'.format(window))
    ax.set_ylabel('Rewards')
    ax.legend()
    
    if create_figure:
        plt.show()
        
# Below are implementations of some standard RL agents
class RandomAgent():
    """Random agent"""
    def __init__(self, env):
        self.env = env
        
    def act(self, state):
        return self.env.action_space.sample()
    
    def reset(self):
        pass
    
    def learn(self, state, action, reward, next_state, done):
        pass
        
class QLearningAgent():
    """
    A Q-learning implementation that uses a dictionary as table
    with flattened observations as keys leveraging gym.spaces.flatten()
    Note: The action space has to be a gym.space.Discrete() object
    """
    def __init__(self, env, gamma, alpha, epsilon_start, epsilon_decay, epsilon_end):
        # Sanity checks related to this particular implementation
        isinstance(env.action_space, spaces.Discrete)
        isinstance(env.observation_space, spaces.Space)
        
        self.env = env
        self.gamma = gamma # Discount factor
        self.alpha = alpha # Learning rate
        self.epsilon_start = epsilon_start # Exploration rate
        self.epsilon_decay = epsilon_decay # Decay after each episode
        self.epsilon_end = epsilon_end # Minimum value
        self.is_greedy = False # Does the agent behave greedily?
        
    def reset(self):
        # Reset Q-table, exploration rate (before training)
        self.q_table = {}
        self.epsilon = self.epsilon_start

    def learn(self, state, action, reward, next_state, done):
        # Compute td-target
        if done:
            td_target = reward # Ignore future return
        else:
            td_target = reward + self.gamma * max(self.get_qvalues(next_state))
            
        # Epsilon decay
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        
        # Update Q-table using the TD target and learning rate
        new_qvalue = (1 - self.alpha) * self.get_qvalues(state)[action] + self.alpha * td_target
        self.get_qvalues(state)[action] = new_qvalue

    def act(self, state):
        # Exploration rate
        epsilon = 0.01 if self.is_greedy else self.epsilon
            
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.get_qvalues(state)) # Greedy action
    
    def get_qvalues(self, state):
        # Flatten state
        state = tuple(spaces.flatten(self.env.observation_space, state))
        
        # Generate new entry in table for new states
        if state not in self.q_table:
            self.q_table[state] = np.random.rand(self.env.action_space.n)
            
        return self.q_table[state]

    def get_epsilons(self, n):
        return self.epsilon_start * (self.epsilon_decay**np.arange(n))
    
    def get_qtable(self, values_fmt='{:.2g}', render=True):
        # Format states and actions
        unflatten_f = lambda x: spaces.unflatten(self.env.observation_space, x)
        states = map(self.env.format_state, map(unflatten_f, self.q_table.keys()))
        actions = map(self.env.format_action, range(self.env.action_space.n))
        
        # Create, format and render DataFrame
        df = pd.DataFrame(self.q_table.values(), list(states), list(actions))
        df = df.applymap(values_fmt.format)
        if render:
            display(df)
        
        return df
