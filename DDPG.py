import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc_dims[0])
        self.fc2 = nn.Linear(self.fc_dims[0], self.fc_dims[1])
        self.mu = nn.Linear(self.fc_dims[1], self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = 0.5 * T.tanh(self.mu(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc_dims[0])
        self.fc2 = nn.Linear(self.fc_dims[0] + self.n_actions, self.fc_dims[1])
        self.fc3 = nn.Linear(self.fc_dims[1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(T.cat([x, action], 1))
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ExplorationNoise(object):
    def __init__(self, mu, sigma=0.001, theta=0.02, dt=1e-4):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.sigma_decay_parameter = 1 - self.dt
        self.x_prev = 0.0

    def gaussian_noise(self, episode, sigma_decay=False):
        if sigma_decay:
            sigma_prime = np.power(0.99, episode) * self.sigma
        else:
            sigma_prime = self.sigma
        x = np.random.normal(self.mu, sigma_prime)
        return x


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, next_state, terminal


class Agent(object):
    def __init__(self, alpha, beta, gamma, input_dims, fc_dims, n_actions, batch_size,
                 tau, device, max_size=50000):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, fc_dims, n_actions=n_actions)
        self.critic = CriticNetwork(beta, input_dims, fc_dims, n_actions=n_actions)

        self.target_actor = ActorNetwork(alpha, input_dims, fc_dims, n_actions=n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, fc_dims, n_actions=n_actions)

        self.device = device

        self.noise = ExplorationNoise(mu=np.zeros(n_actions))

    def choose_action(self, state, episode, turn_off_noise):
        self.actor.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor.forward(state).to(self.device)
        if turn_off_noise:
            mu_prime = mu
        else:
            # mu_prime = mu + T.tensor(self.noise.ou_noise(sigma_decay=True), dtype=T.float).to(self.device)
            mu_prime = mu + T.tensor(self.noise.gaussian_noise(episode, sigma_decay=True), dtype=T.float).to(self.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def soft_update(self, tau):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        q_eval = self.critic.forward(state, action)  # Q(s, a)
        next_action = self.target_actor.forward(next_state)  # a'
        q_next = self.target_critic.forward(next_state, next_action)  # Q(s', a')

        q_target = []
        for j in range(self.batch_size):
            q_target.append(reward[j] + self.gamma * q_next[j] * done[j])
        q_target = T.tensor(q_target).to(self.device)
        q_target = q_target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(q_target, q_eval)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.soft_update(self.tau)
