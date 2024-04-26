from env import rlmc_env
from ddpg import *
from utils import *
import sys


# This file is used to test models
# please change the parameters of env, "max_abs_action", "model_name", and parameters of DDPG
if __name__ == "__main__":
    # env = rlmc_env("N-spring2D", 5, 0.005, "threshold center of grav")  # Creat env
    # filename = sys.argv[1]
    # if filename:
    #     model_name = filename
    # else: 
    #     model_name = "N-spring2D_N=10_dt=0.001_new"
    # state_dim, action_dim = env.NNdims()
    # max_abs_action = 4
    # # Load torch model
    # model = torch.load("pth/" + model_name + ".pth")
    # agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
    #              gamma=0.99, tau=0.005)
    # agent.actor = model
    # print("Simulation Start (from Actor)")
    # env.set_seed(43)
    # episodes = 100
    # steps = 500
    # scores = []
    # for episode in range(episodes):
    #     env.reset_random(5.0)
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
    # env = rlmc_env("N-spring2D", 10, 0.005 , "threshold center of grav")  # Creat env
    # state_dim, action_dim = env.NNdims()
    # print("Simulation Start (from env)")
    # episodes = 100
    # steps = 300
    # scores_env = []
    # env.set_seed(43)
    # for episode in range(episodes):
    #     env.reset_random(5.0)
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
    
    # Animated visulization 
    env = rlmc_env("N-spring2D", 5, 0.005, "threshold center of grav")  # Creat env
    env.set_seed(43)
    state_dim, action_dim = env.NNdims()
    max_abs_action = 4
    filename = sys.argv[1]
    if filename:
        model_name = filename
    else: 
        model_name = "N-spring2D_N=10_dt=0.001_new"
    # Load torch model
    model = torch.load("pth/" + model_name + ".pth")
    agent = DDPG(state_dim, action_dim, max_abs_action, hidden_width0=256, hidden_width1=128, batch_size=256, lr=0.001,
                 gamma=0.99, tau=0.005)
    agent.actor = model
    print("Simulation Start (from Actor)")
    episodes = 1
    steps = 500
    scores = []
    positions = []
    env.reset_random(5.0)
    state = env.get_current_state(n_dt=1)
    score = []
    score_sim = []
    energy = []
    energy_sim = []
    done = False
    simulated_r = []
    # actor
    for step in range(steps):
        action = agent.choose_action(state)
        # print(action)
        next_state, reward, _ = env.step(action, n_dt=1)
        
        if step % 10 == 0:
            energy.append(env.compute_total_U(env.r) + env.compute_total_K(env.v))
            positions.append(env.r) 
            score.append(reward)
        state = next_state
    
    env.reset()

    # simulation 
    for step in range(steps):
        action = env.compute_forces(env.r)
        # print(action)
        next_state, reward, _ = env.step(action, n_dt=1)
        if step % 10 == 0:
            simulated_r.append(env.r) 
            energy_sim.append(env.compute_total_U(env.r) + env.compute_total_K(env.v))
            score_sim.append(reward)
        state = next_state

    # print(np.array(simulated_r).mean(axis = 1))

    visualize(np.array(positions), ['b', 'k', 'r', 'c', 'm'], f"{model_name}_750_actor_vis.gif", (-10, 20), (-10, 20), score, energy)
    visualize(np.array(simulated_r), ['b', 'k', 'r', 'c', 'm'], f"{model_name}_750_sim_vis.gif", (-10, 20), (-10, 20), score_sim, energy_sim)