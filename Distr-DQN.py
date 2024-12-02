import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from maze import Maze

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
        observation, action, reward, next_observation, done = zip(*batch)
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
        self.fc3 = nn.Linear(128, action_dim * atoms_num)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x.view(-1, self.atoms_num), 1).view(-1, self.action_dim, self.atoms_num)
        return x

    def reset_noise(self):
        pass

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
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, atoms_num)

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

def evaluate_agent(env, agent, num_runs=5):
    total_reward = 0
    for _ in range(num_runs):
        state = env.reset_state()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(torch.FloatTensor(np.expand_dims(state, 0)), epsilon=0.0)  # No exploration
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / num_runs
    return average_reward

def plot_path(agent_path, episode):
    grid = np.zeros((env.number_of_tiles, env.number_of_tiles))
    for (row, col) in env.walls:
        grid[row, col] = -1  # Represent walls with -1
    for step, (row, col) in enumerate(agent_path):
        grid[row, col] = step + 1  # Mark path with step number
    grid[env.goal_pos] = 10  # Mark goal with 10
    plt.imshow(grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Steps (0=start, 10=goal)")
    plt.title(f"Path Taken by Agent - Episode {episode}")
    plt.show()

if __name__ == '__main__':
    episode = 1000
    epsilon_init = 0.95
    epsilon_decay = 0.95
    epsilon_min = 0.01
    update_freq = 10
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

    # Initialize lists to store rewards and losses
    episode_rewards = []
    episode_losses = []

    for i in range(episode):
        obs = env.reset_state()
        reward_total = 0
        steps = 0
        cumulative_loss = 0
        agent_path = [obs]  # Track path for visualization
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
                loss = train(eval_net, target_net, buffer, v_min, v_max, atoms_num, gamma, batch_size, optimizer, count, update_freq)
                cumulative_loss += loss
            steps += 1
            agent_path.append(obs)  # Add to path
            if done:
                if epsilon > epsilon_min:
                    epsilon = epsilon * epsilon_decay
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                avg_loss = cumulative_loss / steps if steps > 0 else 0
                episode_rewards.append(reward_total)  # Store total reward
                episode_losses.append(avg_loss)       # Store average loss
                print(f'episode: {i+1}  reward: {reward_total}  weight_reward: {weight_reward:.3f}  epsilon: {epsilon:.2f}')
                break

        # Evaluate the agent and plot path every 50 episodes
        if i % 500 == 0 and i != 0:
            avg_reward = evaluate_agent(env, eval_net)
            print(f"Evaluation after episode {i}: Average Reward = {avg_reward}")
            plot_path(agent_path, i)

    # After training, plot rewards and losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.subplot(1, 2, 2)
    plt.plot(episode_losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()