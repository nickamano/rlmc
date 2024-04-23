from env import rlmc_env
from env_rel import rlmc_env_rel
from ddpg import *
from utils import *
import numpy as np

import signal
import sys

def handler(sig, frame):
    print('Ctrl+C Exit')
    list2txt(scores, model_name)
    plot1(scores, pretrain_episodes, 10, model_name)
    sys.exit(0)

if __name__ == "__main__":
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    """
    Copy this info to test_model.py
    """
    sim_type = "N-spring2D"
    N = 3
    dt_ = 0.005
    reward_type = "initial_energy"
    model_name = "{}_{}_{}_{}".format(sim_type, N, dt_, reward_type)
    """
    end copy
    """

    # env = rlmc_env(sim_type, N, dt_, reward_type)  # Creat env
    env = rlmc_env_rel(sim_type, N, dt_, reward_type)
    print(model_name)
    state_dim, action_dim = env.NNdims()
    max_abs_action = 10
    converge_score = -200
    # Create DDPG Agent
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=state_dim, hidden_width1=state_dim, batch_size=128, lr=0.0005,
                 gamma=0.99, tau=0.002)
    print("Simulation Start")

    episodes = 2000
    pretrain_episodes = 10
    steps = 2000
    scores = []

    rb = ReplayBuffer(state_dim, action_dim)
    for episode in range(episodes + pretrain_episodes):
        env.reset_random(1.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        done = False
        for step in range(steps):
            action = agent.choose_action(env.get_relative_state(n_dt=1))
            next_state, reward, _ = env.step(action, n_dt=1, offline=True, verbose=False)
            # print(reward)
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
        if episode % 50 == 0:
            save_model(agent.actor, model_name + str(episode))


    list2txt(scores, model_name)
    plot1(scores, pretrain_episodes, 10, model_name)    
