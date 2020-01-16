import datetime
import os
import pathlib
import random
from abc import abstractmethod, ABC
from typing import Any
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim

from env import BinaryGridView, DeliveryDrones
from rl_helpers import ReplayMemoryAgent, RandomAgent, MultiAgentTrainer, plot_rolling_rewards, test_agents, \
    plot_cumulative_rewards, DQNAgent


class Logger(ABC):
    def __init__(self, ) -> None:
        super().__init__()
        self.start = datetime.datetime.now()

    @abstractmethod
    def log_dict(self, global_step: int, values: dict) -> None:
        pass


class TensorBoardLogger(Logger):
    def __init__(self, path: str, name: str) -> None:
        super().__init__()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(os.path.join(path, name))

    def log_histogram(self, global_step: int, name: str, values: Any, bins=1000) -> None:
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])
        self.log_tensoboard_summary(global_step, summary)

    def log_dict(self, global_step: int, values: dict) -> None:
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value) for name, value in values.items()])
        self.log_tensoboard_summary(global_step, summary)

    def log_tensoboard_summary(self, global_step: int, summary) -> None:
        self.summary_writer.add_summary(summary, global_step)


class OneHot(nn.Module):
    """ Encodes inputs input one hot vectors
    """
    def __init__(self, depth: int, dtype: torch.dtype = torch.float32):
        """

        Args:
            depth: The size of the one_hot dimension
            dtype: The data type of the output tensor
        """
        super().__init__()
        self.dept = depth
        self.dtype = dtype

    def forward(self, indices):
        x_one_hot = indices.new_zeros(tuple(indices.size()) + (self.dept,))
        x_one_hot.scatter_(dim=-1, index=torch.unsqueeze(indices, dim=-1), src=indices.new_ones(x_one_hot.size()))
        return x_one_hot.to(dtype=self.dtype)


class QNetwork(nn.Module):
    def __init__(self, env, conv_sizes, fc_sizes):
        # Initialize module
        super().__init__()

        # Get input size
        grisize, grisize, depth = env.observation_space.shape

        # Create convolutional layers
        self.conv = nn.Sequential()
        in_channels = depth
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_sizes):
            # Create layer
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

            # Add layer + activation
            self.conv.add_module(f'conv2d_{i + 1}', layer)
            self.conv.add_module(f'ReLU_{i + 1}', nn.ReLU())

            in_channels = out_channels

        # Add classification layer
        self.fc = nn.Sequential()
        self.fc.add_module('flatten', nn.Flatten())

        conv_output = self.conv(torch.ones([1, depth, grisize, grisize]))
        batch_size, flatsize = self.fc(conv_output).shape
        fc_sizes = fc_sizes + [env.action_space.n]
        for i, hidden_size in enumerate(fc_sizes):
            # Create layer
            in_features = flatsize if i == 0 else fc_sizes[i - 1]
            out_features = fc_sizes[i]
            layer = nn.Linear(in_features, out_features)

            # Add layer + activation
            if i > 0:
                self.fc.add_module(f'ReLU_{i + 1}', nn.ReLU())
            self.fc.add_module(f'fc_{i + 1}', layer)

        self.network = nn.Sequential(self.conv, self.fc)

    def forward(self, states: torch.Tensor):
        states = states.permute(0, 3, 1, 2)
        return self.network(states)


class ConvNet(nn.Module):

    def __init__(self, obs_shape: Tuple[int, int, int], conv_sizes: List[Tuple[int, int, int ,int]]) -> None:
        super().__init__()
        self.network = nn.Sequential()
        out_h, out_w, out_c = obs_shape
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_sizes):
            layer = nn.Conv2d(out_c, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.network.add_module(f'conv2d_{i + 1}', layer)
            self.network.add_module(f'ReLU_{i + 1}', nn.ReLU())
            out_h = ((out_h + 2*padding - kernel_size) // stride) + 1
            out_w = ((out_w + 2*padding - kernel_size) // stride) + 1
            out_c = out_channels
        self.out_features = out_h * out_w * out_c

    def forward(self, states: torch.Tensor):
        states = states.permute(0, 3, 1, 2)  # NxHxWxC -> NxCxHxW
        return self.network(states)


class DenseNet(nn.Module):

    def __init__(self, in_features: int, fc_sizes: List[int], out_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential()
        for i, size in enumerate(fc_sizes):
            self.network.add_module(f'fc_{i}', nn.Linear(in_features, size))
            self.network.add_module(f'ReLU_{i}', nn.ReLU())
            in_features = size
        self.network.add_module('fc_out', nn.Linear(in_features, out_features))
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor):
        return self.network(inputs)


class IntrinsicCuriosityModule(nn.Module):

    def __init__(self, obs_shape, num_actions) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.action_one_hot = OneHot(num_actions)
        self.state_embed = ConvNet(obs_shape, [
            (32, 3, 2, 1),
            (32, 3, 2, 1),
            (32, 3, 1, 1),
            (32, 3, 1, 1),
        ])
        self.forward_model = DenseNet(num_actions + self.state_embed.out_features, [256], self.state_embed.out_features)
        self.inverse_model = DenseNet(2*self.state_embed.out_features, [256], num_actions)

    def forward(self, s_t, a_t, s_tp1):
        a_t_one_hot = self.action_one_hot(a_t)
        phi_t = torch.flatten(self.state_embed(s_t), start_dim=1)
        phi_tp1 = torch.flatten(self.state_embed(s_tp1), start_dim=1)
        forward = self.forward_model(torch.cat([a_t_one_hot, phi_t], dim=-1))
        inverse = self.inverse_model(torch.cat([phi_t, phi_tp1], dim=-1))
        return phi_tp1, inverse, forward


class CuriosityDQNAgent(ReplayMemoryAgent):

    def __init__(self, logger, env, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                 target_update_interval, network_fn: callable, eta=0.01, beta=0.2, lmbda=0.1, lr=1e-3):
        # Initialize agent
        super().__init__(env, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size)
        self.logger = logger

        # Set the update interval for the target network
        self.target_update_interval = target_update_interval

        self.network_fn = network_fn
        self.eta = eta
        self.beta = beta
        self.lmbda = lmbda
        self.lr = lr

    def create_qnetwork(self):
        # Create network
        network = self.network_fn(self.env)

        self.icm = IntrinsicCuriosityModule(self.env.observation_space.shape, self.env.action_space.n)

        model = nn.ModuleList([network, self.icm])

        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        return network, optimizer

    def reset(self):
        # Reset agent
        super().reset()

        # Create target network with episode counter
        self.target_network, _ = self.create_qnetwork()
        self.num_episode = 0
        self.episode_reward = 0
        self.total_steps = 0

    def act(self, state):
        # Exploration rate
        epsilon = 0.01 if self.is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.Tensor([state])
            if torch.cuda.is_available():
                state = state.cuda()
            q_values = self.network(state)[0]
            return q_values.argmax().item()  # Greedy action

    def learn(self, state, action, reward, next_state, done):
        self.episode_reward += reward
        self.total_steps += 1
        if done:  # Increment episode counter at the end of each episode
            self.num_episode += 1
            self.logger.log_dict(self.total_steps, {
                'dqn_curiosity/episode_reward': self.episode_reward,
                'dqn_curiosity/memory_size': len(self.memory),
            })
            self.episode_reward = 0

        # Update target network with current one
        if self.num_episode % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        # Epsilon decay
        if done:
            self.epsilons.append(self.epsilon)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            # Sample batch of experience
            batch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)

            if torch.cuda.is_available():
                state = Tensor(state).cuda()
                action = torch.LongTensor(action).cuda()
                reward = Tensor(reward).cuda()
                next_state = Tensor(next_state).cuda()
                done = Tensor(done).cuda()

            # intrinsic reward
            phi_tp1, inverse, forward = self.icm(state, action, next_state)
            reward_intrinsic = self.eta * 0.5 * ((phi_tp1 - forward).pow(2)).sum(1).squeeze().detach()
            reward_extrinsic = reward
            reward += reward_intrinsic

            # Q-value for current state given current action
            q_values = self.network(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_network(next_state)
            next_q_value = next_q_values.max(1)[0]
            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            dqn_loss = (q_value - td_target.detach()).pow(2).mean()

            # Intrinsic curiosity module loss
            inverse_loss = F.cross_entropy(inverse, action.detach())
            forward_loss = 0.5 * ((forward - phi_tp1).pow(2)).mean()
            icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

            loss = self.lmbda * dqn_loss + icm_loss

            self.optimizer.zero_grad()
            icm_loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'dqn_curiosity/loss': loss.data.cpu().numpy(),
                'dqn_curiosity/dqn_loss': dqn_loss.data.cpu().numpy(),
                'dqn_curiosity/inverse_loss': inverse_loss.data.cpu().numpy(),
                'dqn_curiosity/forward_loss': forward_loss.data.cpu().numpy(),
                'dqn_curiosity/icm_loss': icm_loss.data.cpu().numpy(),
                'dqn_curiosity/reward': reward.mean().data.cpu().numpy(),
                'dqn_curiosity/reward_reward_extrinsic': reward_extrinsic.mean().data.cpu().numpy(),
                'dqn_curiosity/reward_intrinsic': reward_intrinsic.mean().data.cpu().numpy(),
            })


class MyDQNAgent(DQNAgent):
    def __init__(self, logger, env, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                 target_update_interval, network_fn: callable):
        super().__init__(env, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                         target_update_interval)
        self.logger = logger
        self.network_fn = network_fn

    def act(self, state):
        # Exploration rate
        epsilon = 0.01 if self.is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.Tensor([state])
            if torch.cuda.is_available():
                state = state.cuda()
            q_values = self.network(state)[0]
            return q_values.argmax().item()  # Greedy action

    def create_qnetwork(self):
        # Create network
        network = self.network_fn(self.env)

        # Move to GPU if available
        if torch.cuda.is_available():
            network.cuda()

        # Create optimizer
        optimizer = optim.Adam(network.parameters())
        return network, optimizer

    def reset(self):
        super().reset()
        self.episode_reward = 0
        self.total_steps = 0

    def learn(self, state, action, reward, next_state, done):
        self.episode_reward += reward
        self.total_steps += 1
        if done:  # Increment episode counter at the end of each episode
            self.num_episode += 1
            self.logger.log_dict(self.total_steps, {
                'dqn/episode_reward': self.episode_reward,
                'dqn/memory_size': len(self.memory),
            })
            self.episode_reward = 0

        # Update target network with current one
        if self.num_episode % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        # Epsilon decay
        if done:
            self.epsilons.append(self.epsilon)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            # Sample batch of experience
            batch = random.sample(self.memory, self.batch_size)
            state, action, reward, next_state, done = zip(*batch)

            if torch.cuda.is_available():
                state = Tensor(state).cuda()
                action = torch.LongTensor(action).cuda()
                reward = Tensor(reward).cuda()
                next_state = Tensor(next_state).cuda()
                done = Tensor(done).cuda()

            # Q-value for current state given current action
            q_values = self.network(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_network(next_state)
            next_q_value = next_q_values.max(1)[0]
            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            loss = (q_value - td_target.detach()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'dqn/loss': loss.data.cpu().numpy(),
            })

def main():
    obs_wrapper = BinaryGridView  # Possible values: CompassQTable, LidarCompassQTable, GridView, BinaryGridView
    num_drones = 5
    env = obs_wrapper(DeliveryDrones(n=num_drones))
    states = env.reset()

    logger = TensorBoardLogger('logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    network_fn = lambda env: QNetwork(env, [(32, 3, 2, 1), (32, 3, 2, 1), (32, 3, 1, 1), (32, 3, 1, 1)], [256])

    cur_agent = CuriosityDQNAgent(logger, env, gamma=0.98, epsilon_start=1, epsilon_decay=0.995, epsilon_end=0.01, memory_size=50000,
                                  batch_size=32, target_update_interval=500, network_fn=network_fn)
    dqn_agent = MyDQNAgent(logger, env, gamma=0.995, epsilon_start=1, epsilon_decay=0.995, epsilon_end=0.01, memory_size=50000,
                           batch_size=32, target_update_interval=500, network_fn=network_fn)

    # Setup opponents
    agents = {drone.index: RandomAgent(env) for drone in env.drones}

    cur_drone = env.drones[0]
    agents[cur_drone.index] = cur_agent
    print('CUR', 'drone_idx=', cur_drone.index)

    dqn_drone = env.drones[1]
    agents[dqn_drone.index] = dqn_agent
    print('DQN', 'drone_idx=', dqn_drone.index)

    # Train for a few steps, plot results
    trainer = MultiAgentTrainer(env, agents, seed=0)
    cur_drone.is_greedy = False
    dqn_drone.is_greedy = False
    trainer.train(100000)
    plot_rolling_rewards(trainer.rewards_log, subset=range(1, num_drones + 1))

    # Test agents
    cur_drone.is_greedy = True
    dqn_drone.is_greedy = True
    rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
    plot_cumulative_rewards(rewards_log, subset=range(1, num_drones + 1))


if __name__ == '__main__':
    main()