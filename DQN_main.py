"""
This script trains a Deep Q-Network (DQN) agent to solve a maze environment. The maze layout, agent, and environment
are defined, and the agent is trained over a specified number of episodes. The script includes functions for evaluating
the agent's performance and plotting the path taken by the agent.
Modules:
    numpy: For numerical operations.
    matplotlib.pyplot: For plotting.
    maze: Custom module for the maze environment.
    DQNAgent: Custom module for the DQN agent.
    rl_evaluator: Custom module for evaluating reinforcement learning agents.
Constants:
    GAME_HEIGHT (int): Height of the game window.
    GAME_WIDTH (int): Width of the game window.
    NUMBER_OF_TILES (int): Number of tiles in the maze.
Maze Layout:
    level (list): List of strings representing the maze layout.
Functions:
    evaluate_agent(env, agent, num_runs=5):
        Evaluates the agent's performance over a specified number of runs.
        Args:
            env (Maze): The maze environment.
            agent (DQNAgent): The DQN agent.
            num_runs (int): Number of runs for evaluation.
        Returns:
            float: Average reward over the evaluation runs.
    plot_path(agent_path, episode):
        Plots the path taken by the agent in the maze.
        Args:
            agent_path (list): List of (row, col) tuples representing the agent's path.
            episode (int): The episode number for the plot title.
Training Loop:
    Trains the DQN agent over a specified number of episodes. Tracks and prints episode details, and plots the agent's
    path and performance metrics (rewards and losses) after training.
"""
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

env = Maze(level, goal_pos=(1, 5), MAZE_HEIGHT=GAME_HEIGHT, MAZE_WIDTH=GAME_WIDTH, SIZE=25)

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
    # Mark start, end, and a few intermediate points
    num_points = min(10, len(agent_path))  # Limit to 10 points
    points_to_plot = [agent_path[i] for i in np.linspace(0, len(agent_path) - 1, num_points, dtype=int)]
    for step, (row, col) in enumerate(points_to_plot):
        grid[row, col] = step + 1  # Mark path with step number
    grid[env.goal_pos] = 10  # Mark goal with 10
    plt.figure(figsize=(env.number_of_tiles, env.number_of_tiles))  # Adjust figure size to match maze size
    plt.imshow(grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Steps (0=start, 10=goal)")
    plt.title(f"Path Taken by Agent - Episode {episode}")
    plt.show()

# Initialize lists to store rewards and losses
episode_rewards = []
episode_losses = []

# Training loop
num_episodes = 150
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
    print(f"Episode {episode} completed with {steps} steps, average loss {avg_loss} and total reward {episode_reward}")
    
    # Evaluate the agent and plot path every 50 episodes
    #if episode % evaluation_interval == 0 and episode != 0:
        #avg_reward = evaluate_agent(env, agent)
        #print(f"Evaluation after episode {episode}: Average Reward = {avg_reward}")
    if episode_reward > -100:
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