import dgl
import copy
import torch
import random
import numpy as np

from collections import deque


class GaussianNoise:

    def __init__(self, action_dim):

        self.action_dim = action_dim
        self.action_min = - 1.0
        self.action_max = + 1.0

        self.epsilon = 0.7
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.000001

    def get_action(self, action):
        noise = np.random.normal(loc=0.0, scale=1.0, size=action.shape)
        new_action = action + self.epsilon * noise

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        return np.clip(new_action, self.action_min, self.action_max)


class OUNoise:

    def __init__(self, mu=0.0, sigma=2.0, theta=1.0, dt=0.05, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayMemoryGraph:

    def __init__(self, length):
        self.memory = deque(maxlen=length)
        self.action_dim = 1

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        state, next_state, action, reward, terminal, valid = transition
        action = torch.tensor(action).reshape(-1, self.action_dim)
        reward = torch.tensor(reward).reshape(-1)
        terminal = torch.tensor(terminal).reshape(-1)
        valid = torch.tensor(valid).reshape(-1)
        self.memory.append((state, next_state, action, reward, terminal, valid))

    def sample(self, size):
        sample = random.sample(self.memory, size)

        state = dgl.batch([i[0] for i in sample])
        next_state = dgl.batch([i[1] for i in sample])
        action = torch.cat([i[2] for i in sample]).reshape(-1, self.action_dim)
        reward = torch.cat([i[3] for i in sample]).reshape(-1)
        terminal = torch.cat([i[4] for i in sample]).reshape(-1)
        valid = torch.cat([i[5] for i in sample]).reshape(-1)

        return state, next_state, action, reward, terminal, valid


class ReplayMemoryGraphPPO:

    def __init__(self, length, action_dim):
        self.max_length = length
        self.memory = deque(maxlen=self.max_length)
        self.action_dim = action_dim

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample_online(self, read_out, length):
        states = []
        for i in range(length):
            states.append(dgl.batch([sample_elem[i] for sample_elem in self.memory]))

        idx = states[-2].ndata['node_type'] == read_out
        reward = states[-2].ndata['reward'][idx].reshape(-1, 1)
        action = states[-2].ndata['action'][idx].reshape(-1, self.action_dim)
        terminal = states[-2].ndata['done'][idx].reshape(-1, 1)
        valid = states[-2].ndata['valid'][idx].reshape(-1, 1) == 1.0

        return states, action, reward, valid, terminal

    def reset(self):
        self.memory = deque(maxlen=self.max_length)


class ReplayBuffer:

    def __init__(self, length):
        self.step = 0
        self.episode_length = length
        self.buffer = {
            'state': [None for _ in range(length + 1)],
            'action': [None for _ in range(length)],
            'reward': [None for _ in range(length)],
            'done': [None for _ in range(length)],
            'valid': [None for _ in range(length)],
        }

    def __len__(self):
        return len(self.memory)

    def warmup(self, obs):
        self.buffer['state'][self.step] = obs

    def insert(self, data):
        state, action, reward, done, valid = data
        self.buffer['state'][self.step + 1] = state
        self.buffer['action'][self.step] = torch.tensor(action)
        self.buffer['reward'][self.step] = torch.tensor(reward)
        self.buffer['done'][self.step] = torch.tensor(done)
        self.buffer['valid'][self.step] = torch.tensor(valid)
        self.step = (self.step + 1) % self.episode_length

    def sample(self):
        state = self.buffer['state'][:self.episode_length]
        next_state = self.buffer['state'][-1]
        action = torch.stack(self.buffer['action']).reshape(self.episode_length, -1)
        reward = torch.stack(self.buffer['reward']).reshape(self.episode_length, -1)
        done = torch.stack(self.buffer['done']).reshape(self.episode_length, -1)
        valid = torch.stack(self.buffer['valid']).reshape(self.episode_length, -1) == 1.0

        # return state, next_state, action, reward, done, valid
        return dgl.batch(state), next_state, action.reshape(-1), reward.reshape(-1), done.reshape(-1), valid.reshape(-1)

    def after_update(self):
        self.buffer['state'][0] = copy.copy(self.buffer['state'][-1])
