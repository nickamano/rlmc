from env import rlmc_env
from Simple_DDPG import Agent
import numpy as np
import torch as T
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = rlmc_env("N-spring2D", 5, 0.005)  # Creat env
    input_dims, n_actions = env.NNdims()
    max_abs_action = 0.75
    # Create DDPG Agent
    agent = Agent(alpha=3e-4, beta=3e-4, gamma=0.99, input_dims=input_dims, fc_dims=[256, 128], n_actions=n_actions,
                  batch_size=128, tau=0.005, clip_action=max_abs_action, max_size=10000,
                  device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
    print("Simulation Start")
    episodes = 200
    steps = 200
    scores = []
    for episode in range(episodes):
        env.reset_random(1.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        done = False
        for step in range(steps):
            # turn_off_noise = False if 0 <= episode < 75 else True
            turn_off_noise = False
            action = agent.choose_action(state, episode, turn_off_noise)
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            if score > -50 and step == steps - 1:
                reward = 100
                done = True
            agent.remember(state, action, reward, next_state, int(done))
            agent.learn()
            state = next_state
        scores.append(score)
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    x = list(range(len(scores)))

    average_scores = []
    ten = []
    for i in range(len(scores)):
        if i < 10:
            ten.append(scores[i])
        else:
            ten[i % 10] = scores[i]
        average_scores.append(np.mean(ten))

    plt.plot(scores, label='score')
    plt.plot(average_scores, label='average score')
    plt.title('plot')
    plt.xlabel('episodes')
    plt.ylabel('score')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()
