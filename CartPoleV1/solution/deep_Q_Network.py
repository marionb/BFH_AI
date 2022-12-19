import gym
# import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# from time import sleep
from datetime import datetime
from dqn_agent import Agent

### 2. Instantiate the Environment and Agent

env = gym.make('CartPole-v1') #, render_mode="human")
#env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=4, action_size=2, seed=0)

# watch an untrained agent
state, _ = env.reset(seed=0)
for j in range(200):
    action = agent.act(state)
    # env.render() # render_mode="rgb_array"
    state, reward, terminated, truncated, info  = env.step(action)
    if terminated or truncated:
        break
# sleep(3)
env.close()


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.0816, eps_decay=0.9837):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    plot_info = list()
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            #
            # If enough samples are available in memory, get random subset and learn
            # if len(agent.memory) > 128:
            #     experiences = agent.memory.sample()
            #     agent.learn(experiences, 0.99)

            done = terminated and truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tepsilon: {:.4f}'.format(i_episode, np.mean(scores_window),eps), end="")
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        plot_info.append([i_episode, np.mean(scores_window), eps])
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 500.0: # v0 = 200 v>v0 = 500
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/checkpoint.pth')
            break
    return scores, plot_info

### 3. Train the Agent with DQN
max_t=1000
eps_start=1.0
eps_end=0.08
eps_decay=0.98
LR = 2e-3
UPDATE_EVERY = 1

scores, plot_info = dqn(n_episodes=800, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

# plot the scores
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

textstr = '\n'.join((
    r'$t_{max}=%.2f$    $LR=%.E$    $update=%.d$' % (max_t,LR,UPDATE_EVERY ),
    r'$\epsilon_{start}=%.2f$    $\epsilon_{end}=%.2f$    $\epsilon_{decay}=%.2f$' % (eps_start,eps_end,eps_decay))
)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
fig.text(0.5, 0.98, textstr, fontsize=14,
        verticalalignment='top', ha='center', bbox=props)

ax1.plot(np.arange(len(scores)), scores)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode #')

plot_info = np.asarray(plot_info)
ax2.plot(plot_info[:,0], plot_info[:,1])
ax2.set_ylabel('Avg. Score')
ax2.set_xlabel('Episode #')

ax3.plot(plot_info[:,0], plot_info[:,2])
ax3.set_ylabel(r'$\epsilon$')
ax3.set_xlabel('Episode #')
# plt.show()
fig.suptitle('')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
# plt.show()
dt = datetime.now()
timestamp=datetime.timestamp(dt)
plt.savefig(f'{timestamp}_test_run.png')

### 4. Watch a Smart Agent!

# agent.qnetwork_local.load_state_dict(torch.load('results/checkpoint.pth'))
#
# env.render_mode = "human"
# for i in range(3):
#     state, _ = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated and truncated
#         if done:
#             break

env.close()