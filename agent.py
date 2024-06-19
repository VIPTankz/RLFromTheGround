import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np
from exploration import EpsilonGreedy
from networks import DeepQNetwork, DuellingDeepQNetwork, FactorizedNoisyLinear
from replay_buffer import NStepReplayBuffer, ReplayBuffer


class DQN:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_steps=100000,
                 fc1_dims=256, fc2_dims=256):
        self.name = "DQN"
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.epsilon = EpsilonGreedy(eps_steps=eps_steps)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.online_net, self.target_net = self.create_nets(fc1_dims, fc2_dims)

        self.update_n_steps = 100

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(max_size=max_mem_size, input_shape=input_dims, n_actions=n_actions,
        self.min_sample_size = 1000

    def create_nets(self, fc1_dims, fc2_dims):
        online_net = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims,
                                  fc2_dims=fc2_dims, device=self.device)

        target_net = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims,
                                  fc2_dims=fc2_dims, device=self.device)

        return online_net, target_net

    def store_transition(self, state, action, reward, state_, terminal):
        self.replay_buffer.store_transition(state, action, reward, state_, terminal)

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value:
            state = T.tensor(np.array([observation / 255]), dtype=T.float32).to(self.device)
            actions = self.online_net.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def compute_target(self, rewards, states_, terminals):
        with T.no_grad():
            q_next_target = self.target_net.forward(states_)
            q_next_target[terminals] = 0.0

            q_target = rewards + self.gamma * T.max(q_next_target, dim=1)[0]
        return q_target

    def learn(self):
        if self.mem_cntr < self.min_sample_size:
            return

        self.optimizer.zero_grad()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        states, actions, rewards, states_, terminals = self.replay_buffer.sample_buffer(self.batch_size)

        q_pred = self.online_net.forward(states)[batch_index, actions]
        q_target = self.compute_target(rewards, states_, terminals)

        loss = self.loss(q_target, q_pred).to(self.device)

        loss.backward()

        T.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)

        self.optimizer.step()

        self.epsilon.decrease()
        self.iter_cntr += 1
        if self.iter_cntr % self.update_n_steps == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


class DoubleDQN(DQN):
    """Double DQN"""

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_steps=100000,
                 fc1_dims=256, fc2_dims=256):
        self.name="doubleDQN"
        DQN.__init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size, eps_steps,
                 fc1_dims=fc1_dims, fc2_dims=fc2_dims)

    def compute_target(self, rewards, states_, terminals):
        with T.no_grad():
            batch_index = T.arange(self.batch_size, dtype=T.int32)
            q_next_target = self.target_net.forward(states_)
            online_action_values = self.online_net.forward(states_)
            q_next_target[terminals] = 0.0

            action_index = T.argmax(online_action_values, dim=1)

            q_target = rewards + self.gamma * q_next_target[batch_index, action_index]

            return q_target


class DDDQN(DoubleDQN):
    """Double Duelling DQN"""

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_steps=100000,
                 fc1_dims=256, fc2_dims=256):

        self.name="duellingDoubleDQN"

        DoubleDQN.__init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size, eps_steps, fc1_dims,
                     fc2_dims)

    def create_nets(self, fc1_dims, fc2_dims):
        target_net = DuellingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims,
                                          fc2_dims=fc2_dims, device=self.device)
        online_net = DuellingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims,
                                          fc2_dims=fc2_dims, device=self.device)

        return online_net, target_net


class NStep3DQN(DDDQN):
    """ double duelling DQN with n-step"""

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,n=3, fc1_dims=256, fc2_dims=256, max_mem_size=100000, eps_steps=100000):
        DDDQN.__init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size, eps_steps, fc1_dims, fc2_dims)
        self.name = "NStep3DQN"
        self.n = n

        self.replay_buffer = NStepReplayBuffer(n=self.n, gamma=gamma, max_size=max_mem_size, input_shape=input_dims, n_actions=n_actions, device=self.device)

    def compute_target(self, rewards, states_, terminals):
        with T.no_grad():
            q_next_target = self.target_net.forward(states_)
            q_next_target[terminals] = 0.0

            q_target = rewards + self.gamma**self.n * T.max(q_next_target, dim=1)[0]
        return q_target

class NoisyNStep3DQN(NStep3DQN):
    """ Nstep 3DQN with noisy nets"""
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,n=3, fc1_dims=256, fc2_dims=256, max_mem_size=100000, eps_steps=100000):
        NStep3DQN.__init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, n=n, fc1_dims=fc1_dims, fc2_dims=fc2_dims, max_mem_size=max_mem_size, eps_steps=eps_steps)

    def create_nets(self, fc1_dims, fc2_dims):
        target_net = DuellingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, device=self.device,linear_layer=FactorizedNoisyLinear)
        online_net = DuellingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, device=self.device, linear_layer=FactorizedNoisyLinear)
        return online_net, target_net

    def choose_action(self, observation):
        with T.no_grad():
            self.online_net.reset_noise()
        return super().choose_action(observation)

    def compute_target(self, rewards, states_, terminals):
        with T.no_grad():
            self.target_net.reset_noise()
        return super().compute_target(rewards,states_,terminals)
