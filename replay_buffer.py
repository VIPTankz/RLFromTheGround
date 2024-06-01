import numpy as np
import torch as T


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, device, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size), dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.device = device

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state/255
        self.new_state_memory[index] = state_/255
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = T.tensor(self.state_memory[batch]).to(self.device)
        actions = T.tensor(self.action_memory[batch]).to(self.device)
        rewards = T.tensor(self.reward_memory[batch]).to(self.device)
        states_ = T.tensor(self.new_state_memory[batch]).to(self.device)
        terminal = T.tensor(self.terminal_memory[batch]).to(self.device)
        return states, actions, rewards, states_, terminal
