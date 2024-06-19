import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

games = ["BattleZone", "NameThisGame", "Phoenix", "Qbert"]
agents = ["NStep3DQN_100k", "DDQN_100k"]


data = {}
for game in games:
    data[game] = {}
    for agent in agents:
        filename = "results\\" + agent + f"\\{agent}_game_{game}.npy"
        try:
            data[game][agent] = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print(f"File {filename} not found.")

def smooth_scores(data, window=100):
    df = pd.DataFrame(data, columns=['score', 'timesteps'])
    df['smoothed_score'] = df['score'].rolling(window, min_periods=1).mean()
    return df['smoothed_score'], df['timesteps']

for game in data:
    for agent in data[game]:
        scores, timesteps = zip(*data[game][agent])
        smoothed_scores, smoothed_timesteps = smooth_scores(list(zip(scores, timesteps)))
        data[game][agent] = (smoothed_scores, smoothed_timesteps)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, game in enumerate(games):
    for agent in agents:
        if agent in data[game]:
            scores, timesteps = data[game][agent]
            axes[i].plot(timesteps, scores, label=agent)
    axes[i].set_title(game)
    axes[i].set_xlabel('Timesteps')
    axes[i].set_ylabel('Smoothed Score')
    axes[i].legend()

plt.tight_layout()
plt.show()
