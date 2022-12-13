import gym
# import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import sleep
from dqn_agent import Agent

### 2. Instantiate the Environment and Agent

env = gym.make('CartPole-v1') # , render_mode="human")
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


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.5, eps_decay=0.9):
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
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated and truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 500.0: # v0 = 200 v>v0 = 500
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '/results/checkpoint.pth')
            break
    return scores

### 3. Train the Agent with DQN


scores = dqn(n_episodes=300)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


### 4. Watch a Smart Agent!

# agent.qnetwork_local.load_state_dict(torch.load('results/checkpoint.pth'))
#
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