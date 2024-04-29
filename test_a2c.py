from env import rlmc_env
from a2c import *
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
    # N = 3
    # dt_ = 0.005
    # reward_type = "threshold_energy"
    # model_name = "{}_{}_{}_{}".format(sim_type, N, dt_, reward_type)
    N = 3
    dt = 0.005
    reward_flag = "center_of_grav"
    # pdb.set_trace()
    # assert reward_flag == "initial_energy" or reward_flag == "threshold_energy" or reward_flag
    model_name = f"{sim_type}_N={N}_dt={dt}_{reward_flag}"
    """
    end copy
    """

    env = rlmc_env(sim_type, N, dt, reward_flag)  # Creat env
    print(model_name)
    state_dim, action_dim = env.NNdims()
    max_abs_action = 15
    converge_score = -200
    actor_params = torch.load("pth_a2c/" + model_name + ".pth")
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        max_abs_action=max_abs_action
    )
    actor.load_state_dict(actor_params)

    print("Simulation Start (from Actor)")
    episodes = 80
    episodes = 80
    steps = 300
    scores = []
    for episode in range(episodes):
        env.reset_random(3.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        done = False
        for step in range(steps):
            action_distribution = actor(torch.FloatTensor(state).unsqueeze(0))
            action = action_distribution.sample().numpy()
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            state = next_state
        scores.append(score)
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    # action from env
    env = rlmc_env("N-spring2D", N, dt, reward_flag)  # Creat env
    state_dim, action_dim = env.NNdims()
    print("Simulation Start (from env)")
    episodes = 80
    steps = 300
    scores_env = []
    for episode in range(episodes):
        env.reset_random(5.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        for step in range(steps):
            action = np.array(env.compute_forces(env.r)).flatten()
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            state = next_state
        scores_env.append(score)
        if len(scores_env) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores_env) / len(scores_env)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores_env[-10:]) / 10))

    import pdb;pdb.set_trace()
    plot2(scores, scores_env, 10, model_name)