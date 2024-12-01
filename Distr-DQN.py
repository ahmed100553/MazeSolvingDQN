import torch
"""
This script implements a Distributional Deep Q-Network (DQN) with Noisy Networks for exploration, 
applied to a maze environment. The main components include:

Classes:
    - replay_buffer: A class for storing and sampling experience tuples.
    - NoisyLinear: A linear layer with added parameter noise for exploration.
    - categorical_dqn: A neural network model for the categorical DQN.

Functions:
    - projection_distribution: Projects the target distribution for the Bellman update.
    - train: Trains the evaluation model using the target model and experience replay buffer.

Main Execution:
    - Initializes the maze environment.
    - Sets hyperparameters for training.
    - Creates instances of the target and evaluation networks, optimizer, and replay buffer.
    - Runs the training loop for a specified number of episodes, performing actions, storing experiences, 
      and training the network.

Hyperparameters:
    - episode: Number of episodes to train.
    - epsilon_init: Initial value of epsilon for epsilon-greedy policy.
    - epsilon_decay: Decay rate of epsilon per episode.
    - epsilon_min: Minimum value of epsilon.
    - update_freq: Frequency of updating the target network.
    - gamma: Discount factor for future rewards.
    - learning_rate: Learning rate for the optimizer.
    - atoms_num: Number of atoms in the categorical distribution.
    - v_min: Minimum value of the support for the distribution.
    - v_max: Maximum value of the support for the distribution.
    - batch_size: Number of samples per batch for training.
    - capacity: Capacity of the replay buffer.
    - exploration: Number of episodes to explore before training.
    - render: Boolean flag to render the environment.

Environment:
    - Maze: A custom maze environment with a specified layout, goal position, and dimensions.
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from maze import Maze  # Import the Maze environment

class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)

        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))

        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_parameter(self):
        mu_range = 1.0 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.output_dim))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class categorical_dqn(nn.Module):
    def __init__(self, observation_dim, action_dim, atoms_num, v_min, v_max):
        super(categorical_dqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.atoms_num = atoms_num
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.noisy1 = NoisyLinear(128, 512)
        self.noisy2 = NoisyLinear(512, self.action_dim * self.atoms_num)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.noisy1(x)
        x = F.relu(x)
        x = self.noisy2(x)
        x = F.softmax(x.view(-1, self.atoms_num), 1).view(-1, self.action_dim, self.atoms_num)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            dist = self.forward(observation)
            dist = dist.detach()
            dist = dist.mul(torch.linspace(self.v_min, self.v_max, self.atoms_num))
            action = dist.sum(2).max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


def projection_distribution(target_model, next_observation, reward, done, v_min, v_max, atoms_num, gamma):
    batch_size = next_observation.size(0)
    delta_z = float(v_max - v_min) / (atoms_num - 1)
    support = torch.linspace(v_min, v_max, atoms_num)

    next_dist = target_model.forward(next_observation).detach().mul(support)
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figs/episodeRewardLossDQN.png}
    \caption{Training performance over 300 episodes. Left: Total reward per episode. Right: Average loss per episode. The reward stabilizes, indicating convergence in cumulative return estimation, while the loss decreases, reflecting reduced prediction error.}
    \label{fig:reward_loss}
\end{figure}


    Tz = reward + (1 - done) * support * gamma
    Tz = Tz.clamp(min=v_min, max=v_max)
    b = (Tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * atoms_num, batch_size).long().unsqueeze(1).expand_as(next_dist)

    proj_dist = torch.zeros_like(next_dist, dtype=torch.float32)
    proj_dist.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
    return proj_dist


def train(eval_model, target_model, buffer, v_min, v_max, atoms_num, gamma, batch_size, optimizer, count, update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    proj_dist = projection_distribution(target_model, next_observation, reward, done, v_min, v_max, atoms_num, gamma)

    dist = eval_model.forward(observation)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, atoms_num)
    dist = dist.gather(1, action).squeeze(1)
    dist.detach().clamp_(0.01, 0.99)
    loss = - (proj_dist * dist.log()).sum(1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eval_model.reset_noise()
    target_model.reset_noise()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    episode = 100000
    epsilon_init = 0.95
    epsilon_decay = 0.95
    epsilon_min = 0.01
    update_freq = 200
    gamma = 0.99
    learning_rate = 1e-3
    atoms_num = 51
    v_min = -10
    v_max = 10
    batch_size = 64
    capacity = 10000
    exploration = 100
    render = False

    # Initialize the maze environment
    level = [
        "XXXXXXXXXXXXX",
        "X           X",
        "X XXX X XXX X",
        "X   X X   X X",
        "XXX X XXX X X",
        "X   X   X   X",
        "X XXX XXX X X",
        "X X   X   X X",
        "X XXX X XXX X",
        "X   X X   X X",
        "XXX XXX XXX X",
        "XP        X X",
        "XXXXXXXXXXXXX",
    ]
    env = Maze(level, goal_pos=(1, 5), MAZE_HEIGHT=360, MAZE_WIDTH=360, SIZE=25)

    action_dim = 4  # Four actions: left, up, right, down
    observation_dim = 2  # Assuming the state is a 2D coordinate (row, col)
    count = 0
    target_net = categorical_dqn(observation_dim, action_dim, atoms_num, v_min, v_max)
    eval_net = categorical_dqn(observation_dim, action_dim, atoms_num, v_min, v_max)
    target_net.load_state_dict(eval_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = replay_buffer(capacity)
    weight_reward = None
    epsilon = epsilon_init

    for i in range(episode):
        obs = env.reset_state()
        reward_total = 0
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            next_obs, reward, done = env.step(action)
            count += 1
            if render:
                env.render()
            buffer.store(obs, action, reward, next_obs, done)
            reward_total += reward
            obs = next_obs
            if i > exploration:
                train(eval_net, target_net, buffer, v_min, v_max, atoms_num, gamma, batch_size, optimizer, count, update_freq)
            if done:
                if epsilon > epsilon_min:
                    epsilon = epsilon * epsilon_decay
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  reward: {}  weight_reward: {:.3f}  epsilon: {:.2f}'.format(i+1, reward_total, weight_reward, epsilon))
                break