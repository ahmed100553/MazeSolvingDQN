import numpy as np
import matplotlib.pyplot as plt  # Add this import
from maze import Maze
from DQNAgent import DQNAgent
from rl_evaluator import RLEvaluator, run_comparison  # Add this import

# Constants
GAME_HEIGHT = 360  # Adjusted for 9x9 maze
GAME_WIDTH = 360   # Adjusted for 9x9 maze
NUMBER_OF_TILES = 9  # Updated for 9x9 maze

# Maze layout
level = [
    "XXXXXXXXXXXXX",
    "X     X     X",
    "X XXX X XXX X",
    "X   X X   X X",
    "XXX X XXX X X",
    "X   X   X   X",
    "X XXX XXX XXX",
    "X X   X   X X",
    "X XXX X XXX X",
    "X   X X   X X",
    "XXX XXX XXX X",
    "XP        X X",
    "XXXXXXXXXXXXX",
]

env = Maze(level, goal_pos=(1, 1), MAZE_HEIGHT=360, MAZE_WIDTH=360, SIZE=9)

# Define state and action dimensions based on your environment's needs
state_dim = 2  # Assuming the state is a 2D coordinate (row, col)
action_dim = 4  # Four actions: left, up, right, down
agent = DQNAgent(state_dim, action_dim)

# Add evaluation function
def evaluate_agent(env, agent, num_runs=5):
    total_reward = 0
    for _ in range(num_runs):
        state = env.reset_state()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state, epsilon=0.0)  # No exploration
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / num_runs
    return average_reward

# Path plotting function
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

# Initialize lists to store rewards and losses
episode_rewards = []
episode_losses = []

# Training loop
num_episodes = 1000
evaluation_interval = 50  # Evaluate every 50 episodes
for episode in range(num_episodes):
    state = env.reset_state()
    done = False
    steps = 0
    cumulative_loss = 0
    episode_reward = 0
    agent_path = [state]  # Track path for visualization

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train()
        cumulative_loss += loss
        episode_reward += reward
        steps += 1
        state = next_state
        agent_path.append(state)  # Add to path

    avg_loss = cumulative_loss / steps if steps > 0 else 0
    episode_rewards.append(episode_reward)  # Store total reward
    episode_losses.append(avg_loss)         # Store average loss

    # Print episode details
    print(f"Episode {episode} completed with {steps} steps and average loss {avg_loss}")
    
    # Evaluate the agent and plot path every 50 episodes
    if episode % evaluation_interval == 0 and episode != 0:
        avg_reward = evaluate_agent(env, agent)
        print(f"Evaluation after episode {episode}: Average Reward = {avg_reward}")
        plot_path(agent_path, episode)

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
