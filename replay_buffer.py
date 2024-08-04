import numpy as np
from collections import deque
import torch as T


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, device, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.device = device
        self.STATE_NORMALIZATION = 255

    def get_state_normalization(self):
        return self.STATE_NORMALIZATION

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state / self.STATE_NORMALIZATION
        self.new_state_memory[index] = state_ / self.STATE_NORMALIZATION

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


class NStepReplayBuffer:
    """replay buffer for n-step learning"""

    def __init__(self, n, gamma, max_size, input_shape, n_actions, device):
        self.n = n
        self.gamma = gamma

        self.state_memory, self.action_memory, self.reward_memory, self.next_state_memory, self.terminal_memory = self.restart()
        self.replay_buffer = ReplayBuffer(max_size, input_shape, n_actions, device)

    def get_state_normalization(self):
        return self.replay_buffer.STATE_NORMALIZATION

    def restart(self):
        state_memory = deque([], maxlen=self.n)
        next_state_memory = deque([], maxlen=self.n)
        action_memory = deque([], maxlen=self.n)
        reward_memory = deque([], maxlen=self.n)
        terminal_memory = deque([], maxlen=self.n)
        return state_memory, action_memory, reward_memory, next_state_memory, terminal_memory

    def sample_buffer(self, batch_size):
        return self.replay_buffer.sample_buffer(batch_size)

    def store_transition(self, state, action, reward, state_, done):
        stored = False
        if len(self.state_memory) == self.n:
            state_0 = self.state_memory[0]
            action_0 = self.action_memory[0]
            done_0 = False
            reward_0 = 0
            for i, (past_reward, past_done) in enumerate(zip(self.reward_memory, self.terminal_memory)):
                reward_0 += (self.gamma ** i) * past_reward
                done_0 = past_done
                if done_0:
                    self.restart()
                    break
            self.replay_buffer.store_transition(state=state_0, action=action_0, reward=reward_0,
                                                state_=self.next_state_memory[-1], done=done_0)
            stored = True

        self.state_memory.append(state)
        self.next_state_memory.append(state_)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)
        return stored

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree():
    def __init__(self, size, procgen=False):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # Initial max value to return (1 = 1^ω)
 
      # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)
 
    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
          self._propagate(parents)
 
    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
          self._propagate_index(parent)
 
    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
 
    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)
 
    def append(self, value):
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)
 
    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
          return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
          children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)
 
  # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices
 
    def total(self):
        return self.sum_tree[0]

class NStepPrioritizedExperienceReplay:
    """Nstep Prioritized experienced replay buffer"""
    def __init__(self, n, gamma, max_size, input_shape, n_actions, device, alpha, beta, epsilon):
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1
        self.device = device
        self.max_size = max_size

        self.state_memory, self.action_memory, self.reward_memory, self.next_state_memory, self.terminal_memory = self.restart()
        self.replay_buffer = NStepReplayBuffer(n=self.n,gamma=self.gamma,max_size=max_size,input_shape=input_shape,n_actions=n_actions,device=device)
        self.sum_tree = SumTree(max_size)

    def get_state_normalization(self):
        return self.replay_buffer.replay_buffer.STATE_NORMALIZATION

    def sample_buffer(self, batch_size):

        segment_length = self.sum_tree.total() / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        samples = np.random.uniform(low=0.0,high=segment_length,size=[batch_size])+segment_starts
        priorities, data_indices, tree_indices = self.sum_tree.find(samples)
        probs = priorities / self.sum_tree.total()
        weights = (self.replay_buffer.replay_buffer.mem_cntr * probs)**(-self.beta)
        weights = weights / weights.max()
        weights = T.tensor(weights, dtype=T.float32, device=self.device)
        return self.sample_by_indices(data_indices), weights, tree_indices

    def sample_by_indices(self, indices):
        states = T.tensor(self.replay_buffer.replay_buffer.state_memory[indices], dtype=T.float32, device=self.device)
        actions = T.tensor(self.replay_buffer.replay_buffer.action_memory[indices], dtype=T.int64, device=self.device)
        next_states = T.tensor(self.replay_buffer.replay_buffer.new_state_memory[indices], dtype=T.float32, device=self.device)
        rewards = T.tensor(self.replay_buffer.replay_buffer.reward_memory[indices], dtype=T.float32, device=self.device)
        dones = T.tensor(self.replay_buffer.replay_buffer.terminal_memory[indices], dtype=T.bool, device=self.device)
        return states, actions, rewards, next_states, dones

    def update_priority(self, tderror, index):
        '''
        After learning 
        '''
        tderror = (tderror + self.epsilon)**self.alpha
        self.max_priority = max(self.max_priority, T.max(tderror).to(self.device))
        self.sum_tree.update(index, tderror)

    def store_transition(self, state, action, reward, state_, done):
        stored = self.replay_buffer.store_transition(state, action, reward, state_, done)
        if stored:
            self.sum_tree.append(self.max_priority)

    def restart(self):
        state_memory = deque([], maxlen=self.n)
        next_state_memory = deque([], maxlen=self.n)
        action_memory = deque([], maxlen=self.n)
        reward_memory = deque([], maxlen=self.n)
        terminal_memory = deque([], maxlen=self.n)
        return state_memory, action_memory, reward_memory, next_state_memory, terminal_memory

if __name__ == "__main__":
    nstepreplay = NStepReplayBuffer(3, gamma=0.99, max_size=10, input_shape=2, n_actions=4, device="cpu")
    nstepreplay.store_transition(state=np.array([1, 1]), action=0, reward=1, state_=np.array([2, 2]), done=False)
    nstepreplay.store_transition(state=np.array([2, 2]), action=0, reward=1, state_=np.array([3, 3]), done=False)
    nstepreplay.store_transition(state=np.array([3, 3]), action=0, reward=1, state_=np.array([4, 4]), done=False)
    nstepreplay.store_transition(state=np.array([4, 4]), action=0, reward=1, state_=np.array([80, 80]), done=True)
    nstepreplay.store_transition(state=np.array([1, 5]), action=0, reward=1, state_=np.array([2, 6]), done=False)
    nstepreplay.store_transition(state=np.array([2, 6]), action=0, reward=1, state_=np.array([3, 7]), done=False)
    nstepreplay.store_transition(state=np.array([3, 7]), action=0, reward=1, state_=np.array([4, 8]), done=False)
    nstepreplay.store_transition(state=np.array([4, 8]), action=0, reward=1, state_=np.array([90, 90]), done=True)
    print(f"state_memory: {nstepreplay.replay_buffer.state_memory}")
    print(f"reward_memory: {nstepreplay.replay_buffer.reward_memory}")
    print(f"new_state_memory: {nstepreplay.replay_buffer.new_state_memory}")
    print(f"action_memory: {nstepreplay.replay_buffer.action_memory}")
    print(f"terminal_memory: {nstepreplay.replay_buffer.terminal_memory}")
