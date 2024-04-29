from env import rlmc_env
from env_rel import rlmc_env_rel
from ddpg import *
from utils import *
import sys

import sys
import matplotlib

# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    # plot2(scores, scores_env, 10, model_name)
    actor_network_episode_number = sys.argv[1]

    """
    Paste info from train_model here
    """
    sim_type = "N-lj2D"
    N = 5
    dt_ = 0.005
    reward_type = "force_only"
    model_name = "{}_{}_{}_{}".format(sim_type, N, dt_, reward_type)
    """
    End Paste
    """

    env_actor = rlmc_env_rel(sim_type, N, dt_, reward_type)  # Creat env
    env_target = rlmc_env(sim_type, N, dt_, reward_type)  # Creat env

    state_dim, action_dim = env_actor.NNdims()
    max_abs_action = 1000
    actor_model_name = "{}{}".format(model_name, actor_network_episode_number)

    # Load torch model
    model = torch.load("pth/" + actor_model_name + ".pth")
    hw0 = int(state_dim * (state_dim - 1)/2)
    hw1 = state_dim
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=hw0, hidden_width1=hw1, batch_size=256, lr=0.005,
                 gamma=0.99, tau=0.002)
    agent.actor = model

    print("Simulation Start (from Actor)")
    episodes = 1
    steps = 3000

    positions_actor = []
    positions_target = []

    actions_actor = []
    actions_target = []
    
    env_target.reset_random(5.0)
    # env_actor.reset_random(5.0)
    env_actor.r = env_target.r
    env_actor.v = env_target.v


    state_actor = env_actor.get_current_state(n_dt=1)
    state_target = env_target.get_current_state(n_dt=1)

    score_actor = 0
    score_target = 0

    done = False
    simulated_r = []
    # actor
    for step in range(steps):
        action_actor = agent.choose_action(env_actor.get_relative_state(env_actor.r))
        # print(state_actor)
        # print("asdf: {}".format(action_actor))
        # print("test: {}".format(agent.choose_action(state_actor + 10*np.ones(state_actor.shape))))
        # print()
        action_target = env_target.compute_forces(env_target.r)

        next_state_actor, reward_actor, _, _ = env_actor.step(action_actor, n_dt=1, step=step, offline="online", verbose=False)
        next_state_target, reward_target, _, _ = env_target.step(action_target, n_dt=1, step=step, offline="online", verbose=False)

        if step % 50 == 0:
            positions_actor.append(env_actor.r)
            positions_target.append(env_target.r)

            actions_actor.append(action_actor)
            actions_target.append(action_target)

            actor_TE = env_actor.compute_total_K(env_actor.v) + env_actor.compute_total_U(env_actor.r)
            target_TE = env_target.compute_total_K(env_target.v) + env_target.compute_total_U(env_target.r)
            # print(actor_TE, target_TE)
            # print(reward_actor, reward_target)
            # print()

        state_actor = next_state_actor
        state_target = next_state_target

    pos_diff = [a - t for a, t in zip(positions_actor, positions_target)]
    act_diff = [a - t.flatten() for a, t in zip(actions_actor, actions_target)]
    # print("pos_diff", pos_diff)
    # print("act_diff", act_diff)


    colormap='gist_rainbow'
    c=np.arange(N)
    visualize(np.array(positions_actor), colormap, c, "{}_actor_{}.gif".format(model_name, actor_network_episode_number))
    visualize(np.array(positions_target), colormap, c, "{}_target_{}.gif".format(model_name, actor_network_episode_number))
