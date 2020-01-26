import random
from collections import defaultdict, Counter, deque

import torch
from torch import Tensor, LongTensor
import numpy as np
import matplotlib.pyplot as plt

from agents.logging import Logger
from .dqn import DQNAgent


class PERAgent(DQNAgent):
    """
    DQN agent using Prioritized Experience Replay
    """

    def __init__(self, env, dqn_factory, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size, target_update_interval, alpha, beta,
                 logger: Logger = None):
        super().__init__(env, dqn_factory, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size, target_update_interval, logger)

        self.priorities = deque(maxlen=memory_size)  # Stores the priorities
        self.alpha = alpha
        self.beta = beta

    def learn(self, state, action, reward, next_state, done):
        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))
        self.episode_reward += reward
        self.total_steps += 1

        if len(self.priorities) > 0:
            max_priority = np.max(self.priorities)
        else:
            max_priority = 1.0
        self.priorities.append(max_priority)

        # End of episode
        if done:
            self.num_episode += 1  # Episode counter
            self.logger.log_dict(self.total_steps, {
                'episode_reward': self.episode_reward,
                'memory_size': len(self.memory),
            })
            self.epsilons.append(self.epsilon)  # Log epsilon value

            # Epislon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            self.episode_reward = 0

        # Periodically update target network with current one
        if self.num_episode % self.target_update_interval == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            prios = np.array(self.priorities)
            probs = prios ** self.alpha
            probs /= probs.sum()

            # Sample batch of experience
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            batch = [self.memory[idx] for idx in indices]
            state, action, reward, next_state, done = zip(*batch)

            # Importance sampling
            total = len(self.memory)
            weights = (total * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            action = LongTensor(action)
            reward = Tensor(reward)
            done = Tensor(done)
            weights = Tensor(weights)

            if torch.cuda.is_available():
                action = action.cuda()
                reward = reward.cuda()
                done = done.cuda()
                weights = weights.cuda()

            # Q-value for current state given current action
            q_values = self.qnetwork(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_qnetwork(next_state)
            next_q_value = next_q_values.max(1)[0]

            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            loss = (q_value - td_target.detach()).abs()

            # We use the individual losses as priorities
            priorities = loss + 1e-5
            for idx, prio in zip(indices, priorities):
                self.priorities[idx] = prio.item()

            # Optimize Q-network as usual
            loss = (loss * weights).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'dqn/loss': loss.data.cpu().numpy(),
                'dqn/reward': reward.mean().data.cpu().numpy(),
            })

    def inspect_priorities(self, top_n=10, max_col=80):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
        plt.plot(self.priorities)
        plt.plot()
        plt.show()
