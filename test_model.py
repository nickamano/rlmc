from env import rlmc_env
from a2c import *
from utils import *


# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    # env = rlmc_env("N-spring2D", 10, 0.001)  # Creat env
    # state_dim, action_dim = env.NNdims()
    # max_abs_action = 4
    # model_name = "N-spring2D_N=10_dt=0.001"
    # # Load torch model
    # model = torch.load("pth/" + model_name + ".pth")
    # agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
    #              gamma=0.99, tau=0.005)
    # agent.actor = model
    # print("Simulation Start (from Actor)")
    # episodes = 100
    # steps = 300
    # scores = []
    # for episode in range(episodes):
    #     env.reset()
    #     state = env.get_current_state(n_dt=1)
    #     score = 0
    #     done = False
    #     for step in range(steps):
    #         action = agent.choose_action(state)
    #         next_state, reward, _ = env.step(action, n_dt=1)
    #         score += reward
    #         state = next_state
    #     scores.append(score)
    #     if len(scores) < 10:
    #         print("Episode {} average score: {}".format(episode, sum(scores) / len(scores)))
    #     else:
    #         print("Episode {} average score: {}".format(episode, sum(scores[-10:]) / 10))

    # # action from env
    # env = rlmc_env("N-spring2D", 10, 0.001)  # Creat env
    # state_dim, action_dim = env.NNdims()
    # print("Simulation Start (from env)")
    # episodes = 100
    # steps = 300
    # scores_env = []
    # for episode in range(episodes):
    #     env.reset_random(1.0)
    #     state = env.get_current_state(n_dt=1)
    #     score = 0
    #     for step in range(steps):
    #         action = np.array(env.compute_forces(env.r)).flatten()
    #         next_state, reward, _ = env.step(action, n_dt=1)
    #         score += reward
    #         state = next_state
    #     scores_env.append(score)
    #     if len(scores_env) < 10:
    #         print("Episode {} average score: {}".format(episode, sum(scores_env) / len(scores_env)))
    #     else:
    #         print("Episode {} average score: {}".format(episode, sum(scores_env[-10:]) / 10))

    # plot2(scores, scores_env, 10, model_name)
    """
    Paste infor from train_model here
    """
    sim_type = "N-spring2D"
    N = 3
    dt_ = 0.005
    reward_type = "threshold_energy"
    model_name = "{}_{}_{}_{}".format(sim_type, N, dt_, reward_type)
    """
    End Paste
    """

    actor_network_episode_number = 100

    env_actor = rlmc_env(sim_type, N, dt_, reward_type)  # Creat env
    env_target = rlmc_env(sim_type, N, dt_, reward_type)  # Creat env

    state_dim, action_dim = env_actor.NNdims()
    max_abs_action = 1000
    actor_model_name = "{}{}".format(model_name, actor_network_episode_number)
    # Load torch model
    model = torch.load("pth/" + actor_model_name + ".pth")
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=state_dim, hidden_width1=state_dim, batch_size=128, lr=0.0005,
                 gamma=0.99, tau=0.002)
    agent.actor = model

    print("Simulation Start (from Actor)")
    episodes = 1
    steps = 2000

    positions_actor = []
    positions_target = []
    
    env_target.reset_random(5.0)
    # env_actor.reset_random(5.0)
    env_actor.r = env_target.r
    env_actor.v = env_target.v


    state_actor = env_actor.get_current_state(n_dt=1)
    state_target = env_target.get_current_state(n_dt=1)

    score_actor = 0
    score_target = 0

    done = False
    for step in range(steps):
        action_actor = agent.choose_action(state_actor)
        action_target = env_target.compute_forces(env_target.r)

        next_state_actor, reward_actor, _ = env_actor.step(action_actor, n_dt=1)
        next_state_target, reward_target, _ = env_target.step(action_target, n_dt=1)

        if step % 20 == 0:
            positions_actor.append(env_actor.r)
            positions_target.append(env_target.r)

        state_actor = next_state_actor
        state_target = next_state_target

    visualize(np.array(positions_actor), ['b', 'k', 'r'], "Nspring_actor{}.gif".format(actor_network_episode_number))
    visualize(np.array(positions_target), ['b', 'k', 'r'], "Nspring_target{}.gif".format(actor_network_episode_number))
