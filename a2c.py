import torch
import torch.nn as nn
import torch.optim as optim
from env import *
import torch.distributions as dist
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, temp=1.0):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, 256)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()
        self.temp = temp

    def forward(self, state):
        x = self.relu(self.fc(state))
        mu = self.fc_mu(x)
        std_pre_softmax = self.fc_std(x)
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


# testenv = rlmc_env("5N-spring2D")
testenv = rlmc_env("N-spring2D", 5, 0.0001)
testenv.set_initial_pos(3 * np.random.rand(testenv.N, testenv.D))
testenv.set_initial_vel(np.zeros((testenv.N, testenv.D)))
testenv.set_initial_energies()

n_dt = 1
actor_state_dim = testenv.N * testenv.D * 2 + 1
critic_state_dim = testenv.N * testenv.D * 2 + 1
action_dim = testenv.N * testenv.D
# import pdb;pdb.set_trace()
actor = Actor(actor_state_dim, action_dim)
critic = Critic(critic_state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
reward_list = []

num_episodes = 3
for episode in range(num_episodes):
    state = testenv.get_current_state(n_dt)
    done = False
    last_reward = None
    iteration_count = 1
    count = 0
    while iteration_count > 0:
        iteration_count += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_distribution = actor(state_tensor)
        # action_distribution = actor(state)
        forces = action_distribution.sample()
        log_probs = action_distribution.log_prob(forces).sum()

        next_state, reward, done = testenv.step(forces.detach().numpy().flatten(), n_dt)
        # import pdb;pdb.set_trace()
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
        print(f"iteration {iteration_count}, reward {reward}")

        # if not last_reward:
        #     last_reward = reward
        
        # else:
        #     """
        #     checking stoping conditions: the difference between current reward and the last reward should be
        #     very small. 
        #     """
        #     if abs(last_reward - reward) < 0.0001:
        #         count += 1
        #         if count >= 100: break
            
        #     else:
        #         count = 0
        if reward > -0.1: break
            
    
    print(f"takes {iteration_count} iterations to converge")
    testenv.reset()

actions = "actions.pth"
torch.save(actor.to("cpu").state_dict(), actions)