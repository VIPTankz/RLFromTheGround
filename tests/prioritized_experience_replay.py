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

def test_sum_tree_total_equals_to_replay_buffer_counter(rb: NStepPrioritizedExperienceReplay):
    '''
    ASSUMPTIONS: The sum tree total should be equal to the number of elements in our nstep RB. Otherwise we might sample zero values from a partially filled buffer.
    GOAL: Ensure the number of elements in the replay buffer equals the sumtree total.
    '''

    state = [0, 2, 1]
    reward = -1
    action = 1
    next_state = [1, 0, 100]
    for _ in range(rb.max_size):
        state[2] = state[2] + 1
        next_state[2] = next_state[2] + 1
        rb.store_transition(state=np.array(state), action=action, reward=reward,state_=np.array(next_state), done=False)
        assert rb.replay_buffer.replay_buffer.mem_cntr == rb.sum_tree.total()

def test_nstep_stores_correct_experiences(rb: NStepPrioritizedExperienceReplay):
    '''
    ASSUMPTIONS:
    GOAL:
    '''
    rb = populate_buffer(4, rb)
    print(rb.replay_buffer.replay_buffer.state_memory[0])
    assert list(rb.replay_buffer.replay_buffer.state_memory[0]) == [0., 2., 2.]
    assert list(rb.replay_buffer.replay_buffer.new_state_memory[0]) == [1., 0., 103.] # should skip 101 and 102
    assert rb.replay_buffer.replay_buffer.new_state_memory[0]==-1-0.99-0.99**2

def test_priorities_are_calculated_correctly_for_large_error(rb: NStepPrioritizedExperienceReplay):
    '''
    ASSUMPTIONS:
    GOAL:
    '''
    assert 1==0
