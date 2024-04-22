from env import rlmc_env
from a2c import *
from utils import *


# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    env = rlmc_env("N-spring2D", 5, 0.001)  # Creat env
    state_dim, action_dim = env.NNdims()
    model_name = "N-spring2D_N=5_dt=0.001"
    # Load torch model
    # model = torch.load("pth/" + model_name + ".pth")
    # agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
    #              gamma=0.99, tau=0.001)
    # agent.actor = model
    actor_params = torch.load(f"pth/{model_name}_a2c.pth")
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    actor.load_state_dict(actor_params)
    print("Simulation Start (from Actor)")
    episodes = 100
    steps = 300
    scores = []
    import pdb;pdb.set_trace()
    for episode in range(episodes):
        env.reset_random(1.0)
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
    env = rlmc_env("N-spring2D", 5, 0.001)  # Creat env
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
            print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
        else:
            print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    plot2(scores, scores_env, 10, model_name)
