import random
from collections import defaultdict, Counter
from collections import deque

import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, LongTensor


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
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon_start = epsilon_start  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay after each episode
        self.epsilon_end = epsilon_end  # Minimum value
        self.is_greedy = False  # Does the agent behave greedily?

    def reset(self):
        # Reset Q-table, exploration rate (before training)
        self.q_table = {}
        self.epsilon = self.epsilon_start
        self.epsilons = []  # Log of epsilon values

    def learn(self, state, action, reward, next_state, done):
        # Compute td-target
        if done:
            td_target = reward  # Ignore future return
        else:
            td_target = reward + self.gamma * max(self.get_qvalues(next_state))

        # Epsilon decay
        if done:
            self.epsilons.append(self.epsilon)
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
            return np.argmax(self.get_qvalues(state))  # Greedy action

    def get_qvalues(self, state):
        # Flatten state
        state = tuple(spaces.flatten(self.env.observation_space, state))

        # Generate new entry in table for new states
        if state not in self.q_table:
            # By adding an entry in the Q-table, we make the agent's
            # behavior dependent on previous runs and hence previous seeds!
            # This is not expected in greedy mode.
            if self.is_greedy:
                return np.random.rand(self.env.action_space.n)
            
            self.q_table[state] = np.random.rand(self.env.action_space.n)

        return self.q_table[state]

    def get_qtable(self, values_fmt='{:.2g}'):
        # Format states
        if hasattr(self.env, 'format_state'):
            unflatten_f = lambda x: spaces.unflatten(self.env.observation_space, x)
            states = map(self.env.format_state, map(unflatten_f, self.q_table.keys()))
        else:
            states = ['state {}'.format(i) for i in range(len(self.q_table))]

        # Format actions
        actions = map(self.env.format_action, range(self.env.action_space.n))

        # Create, format and render DataFrame
        df = pd.DataFrame(self.q_table.values(), list(states), list(actions))
        df = df.applymap(values_fmt.format)

        return df

    def format_state(self, s):
        return self.cardinals[s]
