import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import time

class RLEvaluator:
    def __init__(self, maze_env):
        """
        Initialize the RL algorithm evaluator.
        
        Args:
            maze_env: Maze environment instance
        """
        self.env = maze_env
        self.metrics = defaultdict(list)
        self.episode_histories = defaultdict(list)
        
    def evaluate_value_iteration(self, n_episodes=100, max_steps=1000):
        """
        Evaluate the Value Iteration algorithm.
        """
        print("\nEvaluating Value Iteration...")
        self.env.solve(gamma=0.99, theta=1e-6)  # Run value iteration
        
        for episode in tqdm(range(n_episodes)):
            state = self.env.reset_state()
            total_reward = 0
            steps = 0
            path_taken = [state]
            done = False
            
            while not done and steps < max_steps:
                action = np.argmax(self.env.policy_probs[state])
                next_state, reward, done = self.env.simulate_step(state, action)
                total_reward += reward
                steps += 1
                path_taken.append(next_state)
                state = next_state
            
            self.metrics['VI_rewards'].append(total_reward)
            self.metrics['VI_steps'].append(steps)
            self.metrics['VI_success'].append(done)
            self.episode_histories['VI'].append({
                'total_reward': total_reward,
                'steps': steps,
                'success': done,
                'path': path_taken
            })
    
    def evaluate_sarsa(self, n_episodes=100, max_steps=1000):
        """
        Evaluate the SARSA algorithm.
        """
        print("\nEvaluating SARSA...")
        # Train SARSA
        self.env.sarsa(gamma=0.99, alpha=0.2, epsilon=0.3, episodes=1000)
        
        # Evaluate trained policy
        for episode in tqdm(range(n_episodes)):
            state = self.env.reset_state()
            total_reward = 0
            steps = 0
            path_taken = [state]
            done = False
            
            while not done and steps < max_steps:
                action = self.env.target_policy(state)  # Use target (greedy) policy for evaluation
                next_state, reward, done = self.env.simulate_step(state, action)
                total_reward += reward
                steps += 1
                path_taken.append(next_state)
                state = next_state
            
            self.metrics['SARSA_rewards'].append(total_reward)
            self.metrics['SARSA_steps'].append(steps)
            self.metrics['SARSA_success'].append(done)
            self.episode_histories['SARSA'].append({
                'total_reward': total_reward,
                'steps': steps,
                'success': done,
                'path': path_taken
            })
    
    def plot_learning_curves(self):
        """Plot learning curves comparing algorithm performance."""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(131)
        for alg in ['VI', 'SARSA']:
            rewards = self.metrics[f'{alg}_rewards']
            plt.plot(rewards, label=alg)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        # Plot steps
        plt.subplot(132)
        for alg in ['VI', 'SARSA']:
            steps = self.metrics[f'{alg}_steps']
            plt.plot(steps, label=alg)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        
        # Plot success rate
        plt.subplot(133)
        for alg in ['VI', 'SARSA']:
            success = self.metrics[f'{alg}_success']
            success_rate = [sum(success[:i+1])/(i+1) for i in range(len(success))]
            plt.plot(success_rate, label=alg)
        plt.title('Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_value_comparison(self):
        """Plot and compare value functions between VI and SARSA."""
        plt.figure(figsize=(12, 5))
        
        # Value Iteration state values
        plt.subplot(121)
        vi_values = self.env.state_values.copy()
        mask = np.array([pos in self.env.walls for pos in np.ndindex(vi_values.shape)]).reshape(vi_values.shape)
        sns.heatmap(vi_values, mask=mask, cmap='viridis')
        plt.title('Value Iteration State Values')
        
        # SARSA action values (max over actions)
        plt.subplot(122)
        sarsa_values = np.max(self.env.action_values, axis=2)
        sns.heatmap(sarsa_values, mask=mask, cmap='viridis')
        plt.title('SARSA Max Action Values')
        
        plt.tight_layout()
        plt.show()
    
    def plot_path_heatmap(self, algorithm, episode_idx=-1):
        """Plot heatmap of paths taken by the algorithm."""
        path = self.episode_histories[algorithm][episode_idx]['path']
        grid = np.zeros((self.env.number_of_tiles, self.env.number_of_tiles))
        
        # Mark walls
        for wall in self.env.walls:
            grid[wall] = -1
            
        # Count visits to each cell
        for state in path:
            grid[state] += 1
            
        plt.figure(figsize=(10, 10))
        sns.heatmap(grid, cmap='YlOrRd', 
                   mask=(grid < 0),  # mask walls
                   cbar_kws={'label': 'Visits'})
        plt.title(f'{algorithm} Path Heatmap')
        plt.show()
    
    def generate_statistics(self):
        """Generate and print comprehensive statistics."""
        stats = {}
        for alg in ['VI', 'SARSA']:
            stats[alg] = {
                'avg_reward': np.mean(self.metrics[f'{alg}_rewards']),
                'std_reward': np.std(self.metrics[f'{alg}_rewards']),
                'avg_steps': np.mean(self.metrics[f'{alg}_steps']),
                'std_steps': np.std(self.metrics[f'{alg}_steps']),
                'success_rate': np.mean(self.metrics[f'{alg}_success']),
                'min_steps': np.min(self.metrics[f'{alg}_steps']),
                'max_steps': np.max(self.metrics[f'{alg}_steps'])
            }
        
        return pd.DataFrame(stats).round(2)


def run_comparison(maze_env):
    """Run a complete comparison of Value Iteration and SARSA."""
    evaluator = RLEvaluator(maze_env)
    
    # Run evaluations
    evaluator.evaluate_value_iteration()
    evaluator.evaluate_sarsa()
    
    # Generate plots
    evaluator.plot_learning_curves()
    evaluator.plot_value_comparison()
    
    # Plot path heatmaps
    evaluator.plot_path_heatmap('VI')
    evaluator.plot_path_heatmap('SARSA')

    # Print statistics
    print("\nPerformance Statistics:")
    print(evaluator.generate_statistics())
    
    return evaluator