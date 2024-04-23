from env import rlmc_env
from a2c import *
from utils import *


# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    model = "N-spring2D"
    N = 2
    dt = 0.02
    reward_flag = "threshold_energy"
    model_name = f"{model}_N={N}_dt={dt}_{reward_flag}"
    env = rlmc_env("N-spring2D", N, dt)  # Creat env
    state_dim, action_dim = env.NNdims()
    # Load torch model
    actor_params = torch.load(f"pth/{model_name}_a2c.pth")
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        max_abs_action=0.8
    )
    actor.load_state_dict(actor_params)
    print("Simulation Start (from Actor)")
    episodes = 1000
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
    env = rlmc_env("N-spring2D", N, dt)  # Creat env
    state_dim, action_dim = env.NNdims()
    print("Simulation Start (from env)")
    episodes = 1000
    steps = 300
    scores_env = []
    for episode in range(episodes):
        env.reset_random(3.0)
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
