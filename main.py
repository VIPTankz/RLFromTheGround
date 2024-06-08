from agent import DQN, DoubleDQN, DoubleDuellingDQN, NStepDDDQN
import numpy as np
import time
import torch
import gymnasium as gym
import os
import argparse


def make_env(game):
    return gym.make("ALE/" + game + "-ram-v5")


def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val:
            result.append(f"{arg}_{user_val}")
    return '_'.join(result)


if __name__ == '__main__':

    AGENT_NAME = "NONE"

    parser = argparse.ArgumentParser()

    parser.add_argument('--game', type=str)

    parser.add_argument('--fc1', type=int, default=256)
    parser.add_argument('--fc2', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    game = args.game
    print("Playing Game: " + str(game))

    fc1_dims = args.fc1
    fc2_dims = args.fc2
    lr = args.lr

    env = make_env(game)

    # agent = DQN(gamma=0.99, epsilon=1, lr=0.0008,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000)
    # agent = DoubleDQN(gamma=0.99, epsilon=1, lr=0.0008,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000)

    # agent = DoubleDuellingDQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0],
    #                           batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims,
    #                           fc2_dims=fc2_dims)

    agent = NStepDDDQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0],
                              batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims,
                              fc2_dims=fc2_dims)
    AGENT_NAME = agent.name+"_"+non_default_args(args,parser)

    n_steps = 5000000
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
