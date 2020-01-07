from collections import defaultdict, Counter
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import random
import torch
import gym

# Helper function to set the seeds when needed
def set_seed(env, seed):
    env.seed(seed) # Environment seed
    env.action_space.seed(seed) # Seed for env.action_space.sample()
    np.random.seed(seed) # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed
    random.seed(seed) # seed for Python random library

# Reformat a Q-table for readability
def render_qtable(q_table, action_names={}, fmt='{}'):
    # Create DataFrame
    df = pd.DataFrame(q_table)
    df.columns = [
        (action_names[i] if i in action_names else i)
        for i in range(q_table.shape[1])]
    df.index.name = 'state'
    
    # Format and render
    display(df.applymap(fmt.format))

# A Trainer class
class MultiAgentTrainer():
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
    
# Function to test agents
def test_agents(env, agents, n_steps, seed=None):
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

def plot_cumulated_rewards(rewards_log, ax=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
    
    # Plot rewards
    for name, rewards in rewards_log.items():
        # Work with Numpy array
        rewards = np.array(rewards)
        
        # Compute cumulative sum
        cumsum = np.cumsum(rewards)
        idxs = range(1, len(cumsum) + 1)
        
        # Create label with pickup/crash rate
        label = r'{} (reward: {:.3f}±{:.3f}, pickup: {:.2f}% crash: {:.2f}%)'.format(
            name, np.mean(rewards), np.std(rewards),
            100*np.mean(rewards == 1), 100*np.mean(rewards == -1))
        
        # Plot results
        ax.step(idxs, cumsum, label=label)

    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative reward')
    ax.legend()
    
    if create_figure:
        plt.show()
    
def plot_rolling_rewards(rewards_log, ax=None, window=None, hline=None):
    # Creat figure etc.. if ax none
    create_figure = (ax is None)
    if create_figure:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
        
    for name, rewards in rewards_log.items():
        # Work with Numpy array
        rewards = np.array(rewards)
        steps = range(1, len(rewards)+1)
        
        # Set default for window size
        window = int(len(rewards)/10) if window is None else window
            
        # Plot rolling mean
        rolling_mean = pd.Series(rewards).rolling(window).mean()
        ax.plot(steps, rolling_mean,label=name)
        
    if hline is not None:
        ax.axhline(hline, label='target value', c='C0', linestyle='--')

    # Add title, labels and legend
    ax.set_xlabel('Steps (rolling window: {})'.format(window))
    ax.set_ylabel('Rewards')
    ax.legend()
    
    if create_figure:
        plt.show()

# Base agent
class BaseAgent():
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        self.reset()

    # Reset the agent to its original state
    def reset(self):
        ... # Placeholder: we will need to implement it

    # Pick action based on current state
    def act(self, state):
        return 0 # Placeholder: arbitrarily return action 0

    # Update the Q-table based on an experience
    def learn(self, state, action, reward, next_state, done):
        ... # Placeholder: we will need to implement it
        
# Random agent
class RandomAgent(BaseAgent):
    def act(self, state): # Randomly choose actions
        return np.random.choice(self.action_size)
        
# Basic Q-learning agent (without alpha)
class BasicQLearningAgent(BaseAgent):
    def __init__(self, state_size, action_size, gamma):
        self.gamma = gamma
        super().__init__(state_size, action_size)
    
    def reset(self):
        self.q_table = np.random.rand(self.state_size, self.action_size)

    # Q-learning selects the action with the highest expected return
    def act(self, state):
        return np.argmax(self.q_table[state])

    # Q-learning update rule
    def learn(self, state, action, reward, next_state, done):
        if done: # Ignore future return if this experience is terminal
            self.q_table[state, action] = reward
        else: # Otherwise, update the Q-table using the temporal difference
            self.q_table[state, action] = reward + self.gamma * np.max(self.q_table[next_state])

# Q-learning agent
class QLearningAgent(BasicQLearningAgent):
    def __init__(self, state_size, action_size, gamma, alpha):
        # Adding a "learning rate" parameter
        self.alpha = alpha
        
        # Initialize the Q-table
        super().__init__(state_size, action_size, gamma)

    def learn(self, state, action, reward, next_state, done):
        if done: # Ignore future return if this experience is terminal
            td_target = reward 
        else: # Discounted return
            td_target = reward + self.gamma * np.max(self.q_table[next_state]) 
        
        # Update Q-table using the TD target and learning rate
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * td_target
        
# Q-learning agent with epsilon greedy strategy
class EpsilonGreedyQLearningAgent(QLearningAgent):
    def __init__(self, state_size, action_size, gamma, alpha, epsilon):
        # Add an "exploration rate" parameter
        self.epsilon = epsilon
        
        # Initialize Q-table
        super().__init__(state_size, action_size, gamma, alpha)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size) # Random action
        else:
            return np.argmax(self.q_table[state]) # Greedy action
