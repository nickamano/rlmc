import torch
import torch.nn as nn
import torch.optim as optim
from env import *
import torch.distributions as dist
import torch.nn.functional as F
import os

class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_abs_action, temp=1.0):
            super(Actor, self).__init__()
            self.fc = nn.Linear(state_dim, 256)
            self.relu = nn.ReLU()
            self.fc_mu = nn.Linear(256, action_dim)
            self.fc_std = nn.Linear(256, action_dim)
            self.tanh = nn.Tanh()
            self.temp = temp
            self.max_abs_action = max_abs_action
            

        def forward(self, state):
            x1 = self.relu(self.fc(state))
            x2 = self.max_abs_action * self.relu(self.tanh(x1))
            # x1 = self.relu(self.max_abs_action * self.tanh(state))
            # x2 = self.fc(x1)
            mu = self.fc_mu(x2)
            std_pre_softmax = self.fc_std(x2)
            std = F.softmax(std_pre_softmax / self.temp, dim=-1)
            return dist.Normal(mu, std)

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
        
class A2C(object):
    def __init__(self, model_name, temp, testenv, num_episode=10, max_iterations=1000, max_abs_action=4, n_dt=1):
        self.state_dim = testenv.N * testenv.D * 2 + 1
        self.action_dim = testenv.N * testenv.D
        self.temp = temp
        self.env = testenv
        self.model_name = model_name
        self.actor = Actor(self.state_dim, self.action_dim, max_abs_action,temp)
        self.critic = Critic(self.state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.max_iterations = max_iterations
        self.n_dt = n_dt
        self.num_epidoes = num_episode

    def initialization(self):
        self.env.set_initial_pos(3 * np.random.rand(self.env.N, self.env.D))
        self.env.set_initial_vel(np.zeros((self.env.N, self.env.D)))
        self.env.set_initial_energies()
    
    def train(self):
        scores = []
        for ep in range(self.num_epidoes):
            self.env.reset_random(max_dist=3)
            score = 0
            state = self.env.get_current_state(self.n_dt)
            for iter in range(self.max_iterations):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_distribution = self.actor(state_tensor)
                forces = action_distribution.sample()
                log_probs = action_distribution.log_prob(forces).sum()
                next_state, reward, done = self.env.step(forces.detach().numpy().flatten(), self.n_dt)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)
                td_error = reward + 0.99 * next_value * (1 - int(done)) - value
                critic_loss = td_error.pow(2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                actor_loss = -log_probs * td_error.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                state = next_state
                score += reward
            
            scores.append(score)
            if len(scores) % 5 == 0:
                print(f"average rewards: {sum(scores[-5:])/5} on episode {ep}")
            
            
        
    def save(self):
        if not os.path.exists("pth"):
            os.mkdir("pth")

        path = f"pth/{self.model_name}_a2c.pth"
        torch.save(self.actor.to("cpu").state_dict(), path)

model_name = "N-spring2D"
N = 10
dt = 0.001
reward_flag = "initial energy"
model_full = f"{model_name}_N={N}_dt={dt}_{reward_flag}"
testenv = rlmc_env("N-spring2D", N, dt, reward_flag)
agent = A2C(model_name=model_full, temp=1, testenv=testenv, num_episode=50, max_iterations=300, n_dt=1)
agent.initialization()
agent.train()
agent.save()