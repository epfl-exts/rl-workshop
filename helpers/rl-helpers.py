import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from moviepy.editor import ImageClip, concatenate_videoclips


def set_seed(env, seed):
    """Helper function to set the seeds when needed"""
    env.seed(seed)  # Environment seed
    env.action_space.seed(seed)  # Seed for env.action_space.sample()
    np.random.seed(seed)  # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed
    random.seed(seed)  # seed for Python random library


class MultiAgentTrainer():
    """A class to train agents in a multi-agent environment"""

    def __init__(self, env, agents, reset_agents, seed=None):
        # Save parameters
        self.env, self.agents, self.seed = env, agents, seed

        # Create log of rewards and reset agents
        self.rewards_log = {key: [] for key in self.agents.keys()}
        self.reset(reset_agents)

    def reset(self, reset_agents):
        # Set seed for reproducibility
        if self.seed is not None:
            set_seed(self.env, self.seed)

        # Reset agents and clear log of rewards
        for key, agent in self.agents.items():
            self.rewards_log[key].clear()

            if reset_agents:
                agent.reset()

    def train(self, n_steps):
        # Reset env. and get initial observations
        states = self.env.reset()
        
        # Set greedy flag
        for key, agent in self.agents.items():
            agent.is_greedy = False

        for i in tqdm(range(n_steps), 'Training agents'):
            # Select actions based on current states
            actions = {key: agent.act(states[key]) for key, agent in self.agents.items()}

            # Perform the selected action
            next_states, rewards, dones, _ = self.env.step(actions)

            # Learn from experience
            for key, agent in self.agents.items():
                agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
                self.rewards_log[key].append(rewards[key])
            states = next_states


def test_agents(env, agents, n_steps, seed=None):
    """Function to test agents"""

    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()
    rewards_log = defaultdict(list)
    
    # Set greedy flag
    for key, agent in self.agents.items():
        agent.is_greedy = True

    for _ in tqdm(range(n_steps), 'Testing agents'):
        # Select actions based on current states
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)

        # Save rewards
        for key, reward in rewards.items():
            rewards_log[key].append(reward)

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
    for key, rewards in rewards_log.items():
        if key in subset:
            # Work with Numpy array
            rewards = np.array(rewards)
            pickup = (rewards == 1)
            crashed = (rewards == -1) | (rewards == -2)

            # Compute cumulative sum
            cumsum = np.cumsum(rewards)
            idxs = range(1, len(cumsum) + 1)

            # Create label with pickup/crash rate
            label = r'Drone {} - reward: {:.3f}±{:.3f}, pickup: {:.2f}% ({}) crash: {:.2f}% ({})'.format(
                key, np.mean(rewards), np.std(rewards),
                100 * pickup.mean(), pickup.sum(), 100 * crashed.mean(), crashed.sum())

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

    for key, rewards in rewards_log.items():
        if key in subset:
            # Work with Numpy array
            rewards = np.array(rewards)
            steps = range(1, len(rewards) + 1)
            pickup = (rewards == 1)
            crashed = (rewards == -1) | (rewards == -2)

            # Set default for window size
            window = int(len(rewards) / 10) if window is None else window

            # Plot rolling mean
            rolling_mean = pd.Series(rewards).rolling(window).mean()
            label = 'Drone {} - pickup: {} crash: {}'.format(key, pickup.sum(), crashed.sum())
            ax.plot(steps, rolling_mean, label=label)

    if hline is not None:
        ax.axhline(hline, label='target value', c='C0', linestyle='--')

    # Add title, labels and legend
    ax.set_xlabel('Steps (rolling window: {})'.format(window))
    ax.set_ylabel('Rewards')
    ax.legend()

    if create_figure:
        plt.show()
        
def render_video(env, agents, video_path, n_steps=60, fps=1, seed=None):
    # Initialization
    if seed is not None:
        set_seed(env, seed=seed)
    states = env.reset()

    # Run agents
    frames = []
    for _ in tqdm(range(n_steps), 'Running agents', unit='frame'):
        # Select actions based on current states
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # Perform the selected action
        next_states, rewards, dones, _ = env.step(actions)
        states = next_states
        
        # Save frame
        frames.append(env.render(mode='rgb_array'))

    # Create video
    clips = [ImageClip(frame).set_duration(fps) for frame in frames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_path, fps=24)
