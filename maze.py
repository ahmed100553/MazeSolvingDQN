import numpy as np
import random
from typing import Tuple
from tqdm.auto import tqdm


class Maze:
    def __init__(
        self, level, goal_pos: Tuple[int, int], MAZE_HEIGHT=600, MAZE_WIDTH=600, SIZE=25
    ):
        """
        Maze class to represent a simple maze environment.

        Args:
            level (List[str]): A list of strings representing the maze layout.
            goal_pos (Tuple[int, int]): The goal position (row, col) in the maze.
            MAZE_HEIGHT (int, optional): Height of the maze in pixels. Defaults to 600.
            MAZE_WIDTH (int, optional): Width of the maze in pixels. Defaults to 600.
            SIZE (int, optional): Number of tiles per row/column in the maze. Defaults to 25.
        """
        #self.goal = (23, 20)
        self.goal_pos = goal_pos
        self.number_of_tiles = SIZE
        self.tile_size = MAZE_HEIGHT // self.number_of_tiles
        self.maze, self.walls = self.create_maze(level)
        self.level = level
        self.state = self.get_init_state(self.level)

        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.action_values = np.zeros((self.number_of_tiles, self.number_of_tiles, 4))

    def create_maze(self, level):
        """
        Create a list of positions of walls and maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            List[Tuple[int, int]]: A list of maze positions (row, col) that are not walls.
        """
        maze = []
        walls = []
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == " ":
                    maze.append((row, col))
                elif level[row][col] == "X":
                    walls.append((row, col))
        return maze, walls

    def get_init_state(self, level):
        """
        Get the initial state (player's position) in the maze.

        Args:
            level (List[str]): A list of strings representing the maze layout.

        Returns:
            Tuple[int, int]: The initial state (row, col) in the maze.
        """
        for row in range(len(level)):
            for col in range(len(level[row])):
                if level[row][col] == "P":
                    return (row, col)

    def compute_reward(self, state: Tuple[int, int], action: int):
        """
        Compute the reward for taking an action from the current state.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            float: The reward for taking the action from the current state.
        """
        next_state = self._get_next_state(state, action)

        # If the agent reached the goal
        if next_state == self.goal_pos:
            return 100  # High reward for reaching the goal

        # If the agent hits a wall or stays in the same position
        if next_state == state:
            return -1  # Penalty for hitting a wall

        # Small penalty for each step taken (to encourage shorter paths)
        return -0.5

    def step(self, action):
        """
        Take a step in the maze environment.

        Args:
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action)
        done = next_state == self.goal_pos
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        """
        Simulate a step in the maze environment.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[Tuple[int, int], float, bool]: Tuple containing the next state, reward, and done flag.
        """
        next_state = self._get_next_state(state, action)
        reward = self.compute_reward(state, action)
        done = next_state == self.goal_pos
        return next_state, reward, done

    def _get_next_state(self, state: Tuple[int, int], action: int):
        """
        Get the next state based on the current state and action.

        Args:
            state (Tuple[int, int]): Current state (row, col) in the maze.
            action (int): Action to take (0: left, 1: up, 2: right, 3: down).

        Returns:
            Tuple[int, int]: The next state (row, col) after taking the action.
        """
        if action == 0:  # Move left
            next_state = (state[0], state[1] - 1)
        elif action == 1:  # Move up
            next_state = (state[0] - 1, state[1])
        elif action == 2:  # Move right
            next_state = (state[0], state[1] + 1)
        elif action == 3:  # Move down
            next_state = (state[0] + 1, state[1])
        else:
            raise ValueError("Action value not supported:", action)

        if next_state in self.walls:
            #print(f"Hit wall at {next_state}, staying at {state}")
            return state
        #print(f"Moving from {state} to {next_state} with action {action}")
        return next_state

    def solve(self, gamma=0.99, theta=1e-6):
        """
        Solve the maze environment using the value iteration algorithm.

        Args:
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            theta (float, optional): Threshold for convergence. Defaults to 1e-6.
        """
        delta = float("inf")

        while delta > theta:
            delta = 0
            for row in range(self.number_of_tiles):
                for col in range(self.number_of_tiles):
                    if (row, col) not in self.walls:
                        old_value = self.state_values[row, col]
                        q_max = float("-inf")

                        for action in range(4):
                            next_state, reward, done = self.simulate_step(
                                (row, col), action
                            )
                            value = reward + gamma * self.state_values[next_state]
                            if value > q_max:
                                q_max = value
                                action_probs = np.zeros(shape=(4))
                                action_probs[action] = 1

                        self.state_values[row, col] = q_max
                        self.policy_probs[row, col] = action_probs

                        delta = max(delta, abs(old_value - self.state_values[row, col]))

    def target_policy(self, state):
        av = self.action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))

    def exploratory_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)

        else:
            av = self.action_values[state]
            return np.random.choice(np.flatnonzero(av == av.max()))

    def sarsa(self, gamma=0.99, alpha=0.2, epsilon=0.3, episodes=1000):
        init_state = self.state
        self.action_values = np.zeros((self.number_of_tiles, self.number_of_tiles, 4))
        for _ in tqdm(range(episodes)):
            done = False
            state = init_state
            action = self.exploratory_policy(state, epsilon)
            while not done:
                next_state, reward, done = self.simulate_step(state, action)
                next_action = self.exploratory_policy(next_state, epsilon)
                qsa = self.action_values[state][action]
                next_qsa = self.action_values[next_state][next_action]
                self.action_values[state][action] = qsa + alpha * (
                    reward + gamma * next_qsa - qsa
                )
                state = next_state
                action = next_action

    def reset_goal(self):
        """Reset the goal position"""
        self.state_values = np.zeros((self.number_of_tiles, self.number_of_tiles))
        self.policy_probs = np.full(
            (self.number_of_tiles, self.number_of_tiles, 4), 0.25
        )
        self.goal_pos = random.sample(self.maze, 1)[0]

    def reset_state(self):
        """Reset the maze environment."""
        self.state = self.get_init_state(self.level)

        return self.state