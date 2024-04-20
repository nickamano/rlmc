from env import rlmc_env
from ddpg import *
from utils import *
import numpy as np


if __name__ == "__main__":
    env = rlmc_env("N-spring2D", 5, 0.005)  # Creat env
    state_dim, action_dim = env.NNdims()
    max_abs_action = 0.75
    converge_score = -70
    # Create DDPG Agent
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
                 gamma=0.99, tau=0.005)
    print("Simulation Start")
    episodes = 2000
    pretrain_episodes = 100
    steps = 300
    scores = []
    rb = ReplayBuffer(state_dim, action_dim)
    for episode in range(episodes + pretrain_episodes):
        env.reset_random(1.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        done = False
        for step in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            if score > converge_score and step == steps - 1:
                done = True
            rb.store(state, action, reward, next_state, int(done))
            agent.learn(rb)
            state = next_state
        scores.append(score)
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))
        if episode > 200 and sum(scores[-50:]) / 50 > converge_score:
            break

    save_model(agent.actor, "N-spring2D_N=5_dt=0.005")
    list2txt(scores, "N-spring2D_N=5_dt=0.005")
    plot1(scores, pretrain_episodes, 10, "N-spring2D_N=5_dt=0.005")
