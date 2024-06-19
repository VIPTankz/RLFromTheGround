from agent import DQN, DoubleDQN, DDDQN, NStep3DQN, NoisyNStep3DQN
import numpy as np
import time
import torch
import gymnasium as gym
import os
import argparse
from utils import *


def make_env(game):
    return gym.make("ALE/" + game + "-ram-v5")

def initialize_agent():
    if agent_name=="DQN":
        return DQN(gamma=0.99, epsilon=1, lr=lr,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    elif agent_name=="DDQN":
        return DoubleDQN(gamma=0.99, epsilon=1, lr=lr,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    elif agent_name=="DDDQN":
        return DDDQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0], batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    elif agent_name=="NStep3DQN":
        return NStep3DQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0], batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    elif agent_name=="NoisyNStep3DQN":
        return NoisyNStep3DQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0], batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    return None

if __name__ == '__main__':

    AGENT_NAME = "NONE"

    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str)

    parser.add_argument('--fc1', type=int, default=256)
    parser.add_argument('--fc2', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--agent_name', type=str)

    args = parser.parse_args()

    game = args.game
    print("Playing Game: " + str(game))

    agent_name = args.agent_name
    fc1_dims = args.fc1
    fc2_dims = args.fc2
    lr = args.lr

    env = make_env(game)

    agent = initialize_agent()

    AGENT_NAME = agent_name+"_"+non_default_args(args,parser)

    n_steps = 4000000
    steps = 0
    done = False
    observation, info = env.reset()
    last_time = time.time()
    score = 0
    episode_scores = []
    steps_per_episode = []
    episodes = 0
    last_steps = 1

    while steps < n_steps:

        action = agent.choose_action(observation)

        observation_, reward, done_, trun_, info = env.step(action)
        done_ = np.logical_or(done_, trun_)
        steps += 1

        score += reward

        agent.learn()

        reward = np.clip(reward, -1., 1.)

        agent.store_transition(observation, action, reward, observation_, done_)

        observation = observation_

        if done_:
            episode_scores.append(score)
            steps_per_episode.append(steps)
            score = 0
            env.reset()

        if steps % 1200 == 0 and len(episode_scores) > 0:

            avg_score = np.mean(episode_scores[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'.format(AGENT_NAME, game, avg_score, steps,
                                                        (steps - last_steps) / (time.time() - last_time)), flush=True)
                last_steps = steps
                last_time = time.time()

    print("Finished!")
    episode_scores = np.array(episode_scores)
    steps_per_episode = np.array(steps_per_episode)
    results_combined = np.column_stack((episode_scores, steps_per_episode))
    np.save(AGENT_NAME + ".npy", results_combined)
