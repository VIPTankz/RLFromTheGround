import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from exploration import EpsilonGreedy
from networks import DeepQNetwork
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.epsilon = EpsilonGreedy()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.online_net = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256, device=self.device)

        self.target_net = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256, device=self.device)

        self.update_n_steps = 1000

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(max_size=100000, input_shape=10, n_actions=10 ,device=self.device)
        self.min_sample_size = 1000

    def store_transition(self, state, action, reward, state_, terminal):
        self.replay_buffer.store_transition(state, action, reward, state_, terminal)

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value:
            state = T.tensor([observation]).to(self.device)
            actions = self.online_net.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.min_sample_size:
            return

        self.online_net.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        states, actions, rewards, states_, terminals = self.replay_buffer.sample_buffer(batch)

        q_pred = self.online_net.forward(states)[batch_index, actions]
        with T.no_grad():
            q_next_target = self.target_net.forward(states_)
            q_next_target[terminals] = 0.0

            q_target = rewards + self.gamma*T.max(q_next_target, dim=1)[0]

        loss = self.online_net.loss(q_target, q_pred).to(self.device)

        loss.backward()

        T.nn.utils.clip_grad_norm_(self.online_net.parameters(),10)

        self.optimizer.step()

        self.epsilon.decrease()
        self.iter_cntr += 1
        if self.iter_cntr % self.update_n_steps == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
