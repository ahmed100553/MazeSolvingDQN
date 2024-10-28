from TRPOAgent import TRPOAgent
from maze import Maze
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def train_trpo(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset_state()
        states, actions, rewards, log_probs, values, masks = [], [], [], [], [], []

        # Print initial state for debugging
        print(f"Starting Episode {episode + 1}")
        print(f"Initial State: {state}")

        # Collect trajectories
        while True:
            # Normalize state or flatten if needed (e.g., from (row, col) to [row, col] as a vector)
            flat_state = np.array(state, dtype=np.float32)
            action, log_prob = agent.get_action(flat_state)

            # Take a step in the environment
            next_state, reward, done = env.step(action)
            flat_next_state = np.array(next_state, dtype=np.float32)

            # Log collected data for debugging
            print(f"Action Taken: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

            states.append(flat_state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            masks.append(1 - done)  # mask for non-terminal states

            with torch.no_grad():
                values.append(agent.value_net(torch.FloatTensor(flat_state)).item())

            if done:
                print(f"Episode {episode + 1} Ended with State: {next_state} and Total Reward: {sum(rewards)}")
                break
            state = next_state

        # Convert collected trajectories to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)

        advantages, returns = agent.compute_advantages(rewards, values, masks)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Update the value network
        agent.update_value_net(states, returns)

        # Update the policy network using TRPO update
        agent.update_policy(states, actions, old_log_probs, advantages)

        # Logging episode results for tracking
        print(f"Episode {episode + 1}: Total Reward = {sum(rewards)}")


# Example Maze level layout (modify as needed for your environment)
level = [
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
    "X XXXXXXXX          XXXXX",
    "X XXXXXXXX  XXXXXX  XXXXX",
    "X      XXX  XXXXXX  XXXXX",
    "X      XXX  XXX         X",
    "XXXXXX  XX  XXX        XX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "XXXXXX  XX  XXXXXX  XXXXX",
    "X  XXX      XXXXXXXXXXXXX",
    "X  XXX  XXXXXXXXXXXXXXXXX",
    "X         XXXXXXXXXXXXXXX",
    "X             XXXXXXXXXXX",
    "XXXXXXXXXXX      XXXXX  X",
    "XXXXXXXXXXXXXXX  XXXXX  X",
    "XXXP XXXXXXXXXX         X",
    "XXX                     X",
    "XXX         XXXXXXXXXXXXX",
    "XXXXXXXXXX  XXXXXXXXXXXXX",
    "XXXXXXXXXX              X",
    "XX   XXXXX              X",
    "XX   XXXXXXXXXXXXX  XXXXX",
    "XX    XXXXXXXXXXXX  XXXXX",
    "XX        XXXX          X",
    "XXXX                    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXX",
]

# Create Maze environment
env = Maze(level, goal_pos=(23, 20), MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25)

# Define state and action dimensions based on your environment's needs
state_dim = 2  # Assuming the state is a 2D coordinate (row, col)
action_dim = 4  # Four actions: left, up, right, down

# Instantiate TRPO agent
trpo_agent = TRPOAgent(state_dim=state_dim, action_dim=action_dim)

# Train the TRPO agent in the Maze environment
train_trpo(env, trpo_agent, num_episodes=1000)
