import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions, device):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_vals = self.fc2(x)

        return q_vals


class DuellingDeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)

        self.fc2_advantage = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2_value = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.value_stream = nn.Linear(self.fc2_dims, 1)
        self.advantage_stream = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        x_value = F.relu(self.fc2_value(x))
        x_advantage = F.relu(self.fc2_advantage(x))

        state_value = self.value_stream(x_value)
        advantages = self.advantage_stream(x_advantage)

        return state_value + advantages - T.mean(advantages, dim=1, keepdim=True)
