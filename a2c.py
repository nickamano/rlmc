import torch
import torch.nn as nn
import torch.optim as optim
from env import *
import torch.distributions as dist
import torch.nn.functional as F
import os
import random
import pdb

import signal
import sys

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_abs_action, temp=2.0):
            super(Actor, self).__init__()
            self.fc = nn.Linear(state_dim, 256)
            self.relu = nn.ReLU()
            self.fc_mu = nn.Linear(256, action_dim)
            self.fc_std = nn.Linear(256, action_dim)
            # self.fc1 = nn.Linear(256, 256)
            # self.fc2 = nn.Linear(256, action_dim)
            self.tanh = nn.Tanh()
            self.temp = temp
            self.max_abs_action = max_abs_action
            

        def forward(self, state):
            x1 = self.relu(self.fc(state))
            x2 = self.relu(self.max_abs_action * self.tanh(x1))
            # x1 = self.relu(self.max_abs_action * self.tanh(state))
            # x2 = self.fc(x1)
            mu = self.relu(self.fc_mu(x2))
            std_pre_softmax = self.relu(self.fc_std(x2))
            # pdb.set_trace()
            std = F.softmax(std_pre_softmax)
            return dist.Normal(mu, std)
            # x = self.relu(self.fc(state))
            # x = self.relu(self.fc1(x))
            # return self.max_abs_action * torch.tanh(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.network(state)
        
class A2C:
    def __init__(self, model_name, temp, testenv, num_episode=10, max_iterations=1000, max_abs_action=10, n_dt=1, buffer_capacity=10000, batch_size=64):
        self.env = testenv
        self.state_dim, self.action_dim = testenv.NNdims()
        self.actor = Actor(self.state_dim, self.action_dim, max_abs_action, temp)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.model_name = model_name
        self.num_episodes = num_episode
        self.max_iterations = max_iterations
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.n_dt = n_dt

    def train(self):
        scores = []
        for ep in range(self.num_episodes):
            self.env.reset_random(max_dist=5)
            state = self.env.get_current_state(n_dt=1)
            score = 0
            for iter in range(self.max_iterations):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_distribution = self.actor(state_tensor)
                action = action_distribution.sample()
                # log_probs = action_distribution.log_prob(action).sum(dim=1)

                next_state, reward, _= self.env.step(action.detach().numpy().flatten(), n_dt=self.n_dt, offline=True)
                self.buffer.push(state, action.detach().numpy().flatten(), reward, next_state)

                if len(self.buffer) >= self.batch_size:
                    transitions = self.buffer.sample(self.batch_size)
                    batch = list(zip(*transitions))
                    state_batch, action_batch, reward_batch, next_state_batch = map(torch.FloatTensor, batch)

                    value_batch = self.critic(state_batch)
                    next_value_batch = self.critic(next_state_batch)

                    td_errors = reward_batch + 0.99 * next_value_batch - value_batch
                    critic_loss = td_errors.pow(2).mean()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    action_distribution_batch = self.actor(state_batch)
                    log_probs_batch = action_distribution_batch.log_prob(torch.FloatTensor(action_batch)).sum(dim=1)
                    actor_loss = -(log_probs_batch * td_errors.detach()).mean()
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                state = next_state

            scores.append(score)
            if len(scores) % 10 == 0:
                print(f"average rewards: {sum(scores[-10:])/10} on episode {ep}")

    def save(self):
        if not os.path.exists("pth_a2c"):
            os.mkdir("pth_a2c")
        path = f"pth_a2c/{self.model_name}_{self.num_episodes}.pth"
        torch.save(self.actor.to("cpu").state_dict(), path)


def handler(sig, fram):
    print("control C exit")
    agent.save()
    sys.exit(0)


torch.autograd.set_detect_anomaly(True)
hdl = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, handler)
model_name = "N-spring2D"
N = 3
dt = 0.005
reward_flag = "initial_energy"
# pdb.set_trace()
# assert reward_flag == "initial_energy" or reward_flag == "threshold_energy" or reward_flag
model_full = f"{model_name}_N={N}_dt={dt}_{reward_flag}"
testenv = rlmc_env("N-spring2D", N, dt, reward_flag)
agent = A2C(model_name=model_full, temp=1, testenv=testenv, num_episode=200, max_iterations=300, n_dt=1)

agent.train()
agent.save()