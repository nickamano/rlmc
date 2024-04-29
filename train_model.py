from env import rlmc_env
from env_rel import rlmc_env_rel
from ddpg import *
from utils import *

import signal
import sys
from itertools import permutations
import random

def handler(sig, frame):
    print('Ctrl+C Exit')
    list2txt(scores, model_name)
    plot1(scores, pretrain_episodes, 10, model_name, "scores")
    list2txt(force_diffs, model_name)
    plot1(force_diffs, pretrain_episodes, 10, model_name, "avg_force_diff")  
    sys.exit(0)

if __name__ == "__main__":
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    """
    Copy this info to test_model.py
    """
    sim_type = "N-lj2D"
    N = 5
    dt_ = 0.005
    reward_type = "force_only"
    model_name = "{}_{}_{}_{}".format(sim_type, N, dt_, reward_type)
    """
    end copy
    """

    env_mode = "hybrid"

    # env = rlmc_env(sim_type, N, dt_, reward_type)  # Creat env
    env = rlmc_env_rel(sim_type, N, dt_, reward_type)
    print(model_name)
    state_dim, action_dim = env.NNdims()
    max_abs_action = 1000
    converge_score = -200
    # Create DDPG Agent
    hw0 = int(state_dim * (state_dim - 1)/2)
    hw1 = state_dim
    print(state_dim, hw0, hw1, action_dim)
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=hw0, hidden_width1=hw1, batch_size=512, lr=0.005,
                 gamma=0.99, tau=0.002)
    print("Simulation Start")

    episodes = 20000
    pretrain_episodes = 10
    steps = 2000
    scores = []
    force_diffs = []

    perms = []
    for p in permutations(range(env.N)):
        perms.append(p)

    rb = ReplayBuffer(state_dim, action_dim)
    for episode in range(episodes + pretrain_episodes):
        env.reset_random(5.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        force_diff = 0
        done = False
        for step in range(steps):
            skipfirst = True
            for p in random.sample(perms, N):
                if skipfirst:
                    skipfirst = False
                    pass
                else:
                    r_perm = env.r[p, :]
                    action = agent.choose_action(env.get_relative_state(r_perm))
                    next_state, reward, f_diff, _ = env.fake_step(action, n_dt=1, step=step, offline=env_mode, verbose=False)
                    rb.store(state, action, reward, next_state, int(done))

            action = agent.choose_action(env.get_relative_state(env.r))
            next_state, reward, f_diff, done = env.step(action, n_dt=1, step=step, offline=env_mode, verbose=False)
            rb.store(state, action, reward, next_state, int(done))

            force_diff += f_diff
            score += reward
            # if score > converge_score and step == steps - 1:
            #     done = True

            agent.learn(rb)

            state = next_state

            if done == True:
                break
        print(done, step)
        scores.append(score)
        force_diffs.append(force_diff/(episodes+pretrain_episodes))
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))
        # if episode > 200 and sum(scores[-50:]) / 50 > converge_score:
        #     break
        if episode % 10 == 0: #and not done:
            save_model(agent.actor, model_name + str(episode))

    list2txt(scores, model_name)
    plot1(scores, pretrain_episodes, 10, model_name, "scores")
    list2txt(force_diffs, model_name)
    plot1(force_diffs, pretrain_episodes, 10, model_name, "avg_force_diff")  
