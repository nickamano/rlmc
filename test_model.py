from env import rlmc_env
from ddpg import *
from utils import *


# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    env = rlmc_env("N-spring2D", 10, 0.001)  # Creat env
    state_dim, action_dim = env.NNdims()
    max_abs_action = 4
    model_name = "N-spring2D_N=10_dt=0.001"
    # Load torch model
    model = torch.load("pth/" + model_name + ".pth")
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
                 gamma=0.99, tau=0.005)
    agent.actor = model
    print("Simulation Start (from Actor)")
    episodes = 100
    steps = 300
    scores = []
    for episode in range(episodes):
        env.reset()
        state = env.get_current_state(n_dt=1)
        score = 0
        done = False
        for step in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action, n_dt=1)
            score += reward
            state = next_state
        scores.append(score)
        if len(scores) < 10:
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    # action from env
    env = rlmc_env("N-spring2D", 10, 0.001)  # Creat env
    state_dim, action_dim = env.NNdims()
    print("Simulation Start (from env)")
    episodes = 100
    steps = 300
    scores_env = []
    for episode in range(episodes):
        env.reset_random(1.0)
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

    plot2(scores, scores_env, 10, model_name)
