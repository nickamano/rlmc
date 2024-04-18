from env import rlmc_env
from DDPG import Agent
import numpy as np
import torch as T


if __name__ == "__main__":
    # Creat env
    env = rlmc_env("N-spring2D", 5, 0.005)
    env.set_seed(0)
    env.reset()
    env.set_initial_vel(np.zeros((env.N, env.D)))
    env.set_initial_pos(3 * np.random.rand(env.N, env.D))
    env.set_initial_energies()
    # Create DDPG Agent
    input_dims, n_actions = env.NNdims()
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    agent = Agent(alpha=0.01, beta=0.01, gamma=0.99, input_dims=input_dims, fc_dims=[128, 64], n_actions=n_actions,
                  batch_size=32, tau=0.01, device=device)
    print("Simulation Start")
    episodes = 50
    steps = 20
    for episode in range(episodes):
        score = 0
        done = False
        for step in range(steps):
            turn_off_noise = False if 0 <= episode < 75 else True
            state = env.get_current_state(n_dt=1)
            action_actor = agent.choose_action(state, episode, turn_off_noise)
            action_simu = np.array(env.compute_forces(env.r)).flatten()
            print(action_actor)
            action = action_simu.flatten() + action_actor
            # action = action_actor
            next_state, reward, done = env.step(action, n_dt=1)
            agent.remember(state, action, reward, next_state, int(done))
            state = next_state
            score += reward
        print("Episode {} score: {}".format(episode, score))
        env.reset()
