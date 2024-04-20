import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width0, hidden_width1, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_width0)
        self.fc2 = nn.Linear(hidden_width0, hidden_width1)
        self.fc3 = nn.Linear(hidden_width1, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))  # action clipping
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width0, hidden_width1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_width0)
        self.fc2 = nn.Linear(hidden_width0, hidden_width1)
        self.fc3 = nn.Linear(hidden_width1, 1)

    def forward(self, state, action):
        q = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float)
        self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float)
        self.r = torch.zeros((self.max_size, 1), dtype=torch.float)
        self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float)
        self.dw = torch.zeros((self.max_size, 1), dtype=torch.float)

    def store(self, s, a, r, s_, dw):
        index = self.count % self.max_size
        self.s[index] = torch.tensor(s, dtype=torch.float)
        self.a[index] = torch.tensor(a, dtype=torch.float)
        self.r[index] = torch.tensor(r, dtype=torch.float)
        self.s_[index] = torch.tensor(s_, dtype=torch.float)
        self.dw[index] = torch.tensor(dw, dtype=torch.float)
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        batch_s = self.s[indices]
        batch_a = self.a[indices]
        batch_r = self.r[indices]
        batch_s_ = self.s_[indices]
        batch_dw = self.dw[indices]
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_width0, hidden_width1, batch_size, lr, gamma, tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.batch_size = batch_size
        self.lr = lr
        self.GAMMA = gamma
        self.TAU = tau

        self.actor = Actor(state_dim, action_dim, hidden_width0, hidden_width1, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, hidden_width0, hidden_width1)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s)
        return a.detach().cpu().numpy().flatten()  # only convert to numpy if necessary

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for param in self.critic.parameters():
            param.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
