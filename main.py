from agent import DQN, DoubleDQN, DoubleDuellingDQN
import numpy as np
import time
import torch
import gymnasium as gym
import os
import argparse


def make_env(game):
    return gym.make("ALE/" + game + "-ram-v5")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # games = ["BattleZone", "NameThisGame", "Phoenix", "Qbert"]

    parser.add_argument('--game', type=str, default="BattleZone")

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

    agent_name = "Agent"
    # agent = DQN(gamma=0.99, epsilon=1, lr=0.0008,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000)
    # agent = DoubleDQN(gamma=0.99, epsilon=1, lr=0.0008,input_dims=env.observation_space.shape[0],batch_size=32,n_actions=env.action_space.n,max_mem_size=1000000)

    agent = DoubleDuellingDQN(gamma=0.99, epsilon=1, lr=lr, input_dims=env.observation_space.shape[0],
                              batch_size=32, n_actions=env.action_space.n, max_mem_size=1000000, fc1_dims=fc1_dims,
                              fc2_dims=fc2_dims)

    n_steps = 5000000
    steps = 0
    done = False
    observation, info = env.reset()
    last_time = time.time()
    score = 0
    scores = []
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
            scores.append(score)
            score = 0
            env.reset()

        if steps % 1200 == 0 and len(scores) > 0:

            avg_score = np.mean(scores[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'.format(agent_name, game, avg_score, steps,
                                                        (steps - last_steps) / (time.time() - last_time)), flush=True)
                last_steps = steps
                last_time = time.time()

    print("Finished!")
    np.save(agent_name + ".npy", np.array(scores))
