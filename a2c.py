import torch
import torch.nn as nn
import torch.optim as optim
from env import *
import torch.distributions as dist

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, 256)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc(state))
        mu = self.fc_mu(x)
        std = torch.exp(self.fc_std(x))  # log std -> std
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


env = rlmc_env("5N-spring2D")
state_dim = env.N * env.D * 2 
action_dim = env.N * env.D 
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

num_episodes = 20
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_distribution = actor(state_tensor)
        forces = action_distribution.sample()
        log_probs = action_distribution.log_prob(forces).sum()

        next_state, reward, done = env.step(forces.detach().numpy().flatten())

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        value = critic(state_tensor)
        next_value = critic(next_state_tensor)
        td_error = reward + 0.99 * next_value * (1 - int(done)) - value

        critic_loss = td_error.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss = -log_probs * td_error.detach()  # Use detached TD error for loss calculation
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
    
    if episode % 5 == 0:
        print(f'Episode {episode}: Done')
