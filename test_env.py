from env import rlmc_env
from ddpg import *
from utils import *
import numpy as np


if __name__ == "__main__":
    env = rlmc_env("N-spring2D", 5, 0.005)  # Creat env
    state_dim, action_dim = env.NNdims()
    max_abs_action = 0.75
    converge_score = -100
    model_name = "N-spring2D_N=5_dt=0.005"
    print("Simulation Start")
    episodes = 100
    steps = 300
    scores = []
    max_action = 0
    min_action = 0
    for episode in range(episodes):
        env.reset_random(1.0)
        state = env.get_current_state(n_dt=1)
        score = 0
        for step in range(steps):
            action = np.array(env.compute_forces(env.r)).flatten()
            min_action = min(min_action, min(action))
            max_action = max(max_action, max(action))
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            state = next_state
        scores.append(score)
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    print(min_action, max_action)
