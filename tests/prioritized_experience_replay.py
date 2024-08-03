import pytest
import torch as T
import random
import numpy as np
from pathlib import Path
import sys

from replay_buffer import NStepPrioritizedExperienceReplay
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)


@pytest.fixture
def rb(mocker):
    # mocker.patch("package.module.ClassName.function_name", function_that_mocks_it)

    GAMMA = 0.99
    MAX_REPLAY_SIZE = 32
    DEVICE = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    prioritized_replay_buffer = NStepPrioritizedExperienceReplay(n=3,gamma=GAMMA,max_size=MAX_REPLAY_SIZE,input_shape=3, n_actions=4, device=DEVICE,alpha=0.5,beta=0.45,epsilon=0.000001)
    prioritized_replay_buffer.replay_buffer.replay_buffer.STATE_NORMALIZATION = 1

    return prioritized_replay_buffer

def populate_buffer(num, buffer: NStepPrioritizedExperienceReplay):
    '''
    add a number of records into the replay buffer
    '''
    state = [0, 2, 1]
    reward = -1
    action = 1
    next_state = [1, 0, 100]
    for _ in range(num):
        state[2] = state[2] + 1
        next_state[2] = next_state[2] + 1
        buffer.store_transition(state=np.array(state), action=action, reward=reward,state_=np.array(next_state), done=False)
    return buffer

def test_(rb: NStepPrioritizedExperienceReplay):
    '''
    ASSUMPTIONS:
    GOAL:
    '''

    rb = populate_buffer(20, rb)
    print(rb.replay_buffer.replay_buffer.state_memory)
    print(rb.replay_buffer.replay_buffer.reward_memory)
    print(rb.replay_buffer.replay_buffer.new_state_memory)
        # self.sample_by_indices(data_indices), weights, tree_indices
    print(rb.sample_buffer(batch_size=2)[0])
    print(f"******")
    print(rb.sample_buffer(batch_size=2)[1])
    print(f"******")
    print(rb.sample_buffer(batch_size=2)[2])

    assert 1==0
