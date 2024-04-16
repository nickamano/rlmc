from env import rlmc_env
from DDPG import Agent
import numpy as np
import torch as T


if __name__ == "__main__":
    # Creat env
    env = rlmc_env("N-spring2D", 5)
    env.set_seed(0)
    env.reset()
    env.set_initial_vel(np.zeros((env.N, env.D)))
    env.set_initial_pos(3 * np.random.rand(env.N, env.D))
    env.set_initial_energies()
    # Create DDPG Agent
    input_dims, n_actions = env.NNdims()
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    agent = Agent(alpha=0.001, beta=0.001, gamma=0.99, input_dims=input_dims, fc_dims=[128, 64], n_actions=n_actions,
                  batch_size=64, tau=0.001, device=device)
    print("Simulation Start")
    episodes = 100
    for episode in range(episodes):
        score = 0
        done = False
        while not done:
            turn_off_noise = False if 0 <= episode < 75 else True
            state = np.concatenate((env.v, env.r)).flatten()
            action = agent.choose_action(state, episode, turn_off_noise)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, int(done))
            state = next_state
            score += reward
        print("Step {} reward: {}".format(episode, score))
    env.reset()

    # for i in range(1000):
    #     state = np.concatenate((env.v, env.r)).flatten()
    #     print(state)
    #     action = env.compute_forces()  # Replace this action with the action from the actor network
    #     next_state, reward, done = env.step(action)
    #     print("Step {} reward: {}".format(i, reward))
    #     print(next_state)
    # env.reset()
