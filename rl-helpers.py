from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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

# Print Q-table as a DataFrame
def format_qtable(q_table, action_names={}, fmt='{}'):
    df = pd.DataFrame(q_table)
    df.columns = [(action_names[i] if i in action_names else i) for i in range(q_table.shape[1])]
    df.index.name = 'state'
    return df.applymap(fmt.format)
    
# Function to plot values related to training an agent ex. returns
def plot_values(values, name, ax, rolling_window=None, hline=None):
    if len(values) > 0:
        episodes_range = range(1, len(values)+1) # One value per episode
        
        if rolling_window is None:
            ax.plot(episodes_range, values, label=name, c='C1')
        
        else:
            ax.plot(episodes_range, values, alpha=0.2, c='C0') # Plot values
            
            # Rolling mean and standard deviation
            rolling_mean = pd.Series(values).rolling(rolling_window).mean()
            rolling_std = pd.Series(values).rolling(rolling_window).std()

            # Plot rolling mean ± 1 std
            ax.fill_between(
                range(1, len(rolling_mean)+1),
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2, facecolor='C0'
            )
            ax.plot(
                range(1, len(rolling_mean)+1),
                rolling_mean, c='C1',
                label='rolling mean ± std (n={})'.format(rolling_window)
            )
        
    if hline is not None:
        ax.axhline(hline, label='target value', c='C0', linestyle='--')

    # Add title, labels and legend
    last_value = 'None' if len(values) == 0 else '{:.4f}'.format(values[-1])
    ax.set_title('Episodes: {}, last value: {}'.format(len(values), last_value))
    ax.set_xlabel('Episode')
    ax.set_ylabel(name)
    ax.legend()
    
# Function to print results
def print_results(agent_logs):
    print('Results on {:,} episodes'.format(len(agent_logs['returns'])))
    print('Mean episodes return: {:.3f}, std: {:.3f}'.format(
        np.mean(agent_logs['returns']), np.std(agent_logs['returns'])))
    print('Mean episodes length: {:.1f}, std: {:.1f}'.format(
        np.mean(agent_logs['lengths']), np.std(agent_logs['lengths'])))

# A Trainer class
class MultiAgentTrainer():
    def __init__(self, env, agents, seed):
        # Save parameters
        self.env, self.agents, self.seed = env, agents, seed
        
        # Create log of rewards and reset agents
        self.all_rewards = {name: [] for name in self.agents.keys()}
        self.reset()
        
    def reset(self):
        # Set seed for reproducibility
        set_seed(self.env, self.seed)
        
        # Reset agents and clear log of rewards
        for name, agent in self.agents.items():
            agent.reset()
            self.all_rewards[name].clear()

    def train(self, n_steps):
        # Reset env. and get initial observations
        states = self.env.reset()
        
        for i in range(n_steps):
            # Select actions based on current states
            actions = {name: agent.act(states[name]) for name, agent in self.agents.items()}

            # Perform the selected action
            next_states, rewards, dones, _ = self.env.step(actions)

            # Learn from experience
            for name, agent in self.agents.items():
                agent.learn(states[name], actions[name], rewards[name], next_states[name], dones[name])
                self.all_rewards[name].append(rewards[name])
                states = next_states
    
# Function to test agents
def test_agent(env, agent, nb_evaluations=1, seed=0, action_names=None, max_nb_timesteps=None, verbose=False, render=False):
    # Set defaults for optional parameters
    if action_names is None:
        action_names = range(env.action_space.n) # Use 0..n action number as name
    if max_nb_timesteps is None:
        max_nb_timesteps = env.env.spec.max_episode_steps # Use default env. value
    
    # Dictionary to store episode lengths and returns
    episodes_log = {'lengths': [], 'returns': []}
    
    # Set seeds
    set_seed(env, seed=seed)
    
    # Run agent for the specified number of episodes
    for i_episode in tqdm(range(nb_evaluations)):
        # Initialize episode
        episode_return, episode_length, done = 0, 0, False # Initialization
        state = env.reset() # Get initial observation
        
        # Render and print initial state when needed
        if render:
            env.render()
        if verbose:
            print('initial state: {:<2}'.format(state))
                
        # Run the agent until it reaches the goal or the max. number of timesteps
        while not done and episode_length < max_nb_timesteps:
            # Select and perform next action
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1

            # Render and print progress if needed
            if render:
                env.render()
            if verbose:
                print('step: {}, state: {}, action: {}, next state: {}, reward: {}'.format(
                    episode_length, state, action_names[action], next_state, reward))

            state = next_state

        # End of episode: log length and return
        if verbose:
            print('Agent finished evaluation {} after {} timesteps with a return of {:.2f}'.format(
                i_episode+1, episode_length, episode_return))

        episodes_log['returns'].append(episode_return)
        episodes_log['lengths'].append(episode_length)

    return episodes_log

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
