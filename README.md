# Maze Solver with Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) agent to solve a maze environment. The agent is trained to navigate through the maze to reach a goal position, using reinforcement learning techniques.

## Project Structure

- `maze.py`: Defines the `Maze` class, representing the maze environment.
- `DQNAgent.py`: Defines the `DQNAgent` class, implementing the DQN algorithm.
- `dqn_main.py`: Main script to train the DQN agent and evaluate its performance.
- `dqn2.ipynb`: Jupyter notebook for training and visualizing the DQN agent's performance.
- `rl_evaluator.py`: Custom module for evaluating reinforcement learning agents.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/maze-dqn.git
    cd maze-dqn
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the DQN Agent

To train the DQN agent, run the `dqn_main.py` script:
```bash
python [dqn_main.py]
```
GitHub Copilot
Sure! Here is a README.md file for your project:

This script will train the agent over a specified number of episodes and plot the rewards and losses after training.

Evaluating the Agent
The evaluate_agent function in dqn_main.py can be used to evaluate the agent's performance over a specified number of runs. The function returns the average reward over the evaluation runs.

Visualizing the Agent's Path
The plot_path function in dqn_main.py can be used to visualize the path taken by the agent in the maze. The function plots the start, end, and a few intermediate points along the agent's path.

Jupyter Notebook
The dqn2.ipynb notebook provides an interactive environment for training and visualizing the DQN agent's performance. You can run the notebook to see detailed plots and animations of the agent's learning process.

Maze Environment
The maze environment is defined in maze.py. The Maze class provides methods for creating the maze, resetting the state, taking steps, and computing rewards.

### Maze Level

The maze level is defined as follows:
```python
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
```