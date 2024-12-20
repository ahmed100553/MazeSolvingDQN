import pygame
import numpy as np
from maze import Maze
import threading
from rl_evaluator import RLEvaluator, run_comparison  # Add this import


# Constants
GAME_HEIGHT = 600
GAME_WIDTH = 600
NUMBER_OF_TILES = 25
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 700
TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES

# Maze layout
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

env = Maze(
    level,
    goal_pos=(20, 19),
    MAZE_HEIGHT=GAME_HEIGHT,
    MAZE_WIDTH=GAME_WIDTH,
    SIZE=NUMBER_OF_TILES,
)

# Add evaluation before starting the game loop
print("Running algorithm evaluation...")
evaluator = run_comparison(env)
vi_rewards = evaluator.metrics['VI_rewards']
sarsa_rewards = evaluator.metrics['SARSA_rewards']

# Save evaluation results if needed
#np.save('vi_rewards.npy', vi_rewards)
#np.save('sarsa_rewards.npy', sarsa_rewards)

# Initialize Pygame and continue with your existing game loop...
pygame.init()
screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
pygame.display.set_caption("Maze Solver with Evaluation")

env.solve()


TILE_SIZE = GAME_HEIGHT // NUMBER_OF_TILES


# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
pygame.display.set_caption("Maze Solver")  # Set a window title

surface = pygame.Surface((GAME_HEIGHT, GAME_WIDTH))
clock = pygame.time.Clock()
running = True

# Get the initial player and goal positions
treasure_pos = env.goal_pos
player_pos = env.state


def reset_goal():
    # Check if the player reached the goal, then reset the goal
    if env.state == env.goal_pos:
        env.reset_goal()
        env.solve()
current_algorithm = 'VI'  # Start with Value Iteration

# Game loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle between algorithms
                current_algorithm = 'SARSA' if current_algorithm == 'VI' else 'VI'
                print(f"Switched to {current_algorithm}")
            elif event.key == pygame.K_r:
                # Reset the environment
                env.reset_state()
                player_pos = env.state
  
    # Use the current algorithm for action selection
    if current_algorithm == 'VI':
        action = np.argmax(env.policy_probs[player_pos])
    else:  # SARSA
        action = env.target_policy(player_pos)
    # Start a new thread
    x = threading.Thread(target=reset_goal)
    x.daemon = True
    x.start()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the surface
    surface.fill((27, 64, 121))

    # Draw the walls in the maze
    for row in range(len(level)):
        for col in range(len(level[row])):
            if level[row][col] == "X":
                pygame.draw.rect(
                    surface,
                    (241, 162, 8),
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

    # Draw the player's position
    pygame.draw.rect(
        surface,
        (255, 51, 102),
        pygame.Rect(
            player_pos[1] * TILE_SIZE,
            player_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=3,
    )

    # Draw the goal position
    pygame.draw.rect(
        surface,
        "green",
        pygame.Rect(
            env.goal_pos[1] * TILE_SIZE,
            env.goal_pos[0] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        ).inflate(-TILE_SIZE / 3, -TILE_SIZE / 3),
        border_radius=TILE_SIZE,
    )

    # Update the screen
    screen.blit(
        surface, ((SCREEN_HEIGHT - GAME_HEIGHT) / 2, (SCREEN_WIDTH - GAME_WIDTH) / 2)
    )
    pygame.display.flip()

    # Get the action based on the current policy
    action = np.argmax(env.policy_probs[player_pos])
    # action = np.argmax(env.action_values[player_pos])

    # Move the player based on the action
    if (
        action == 1
        and player_pos[0] > 0
        and (player_pos[0] - 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] - 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 3
        and player_pos[0] < NUMBER_OF_TILES - 1
        and (player_pos[0] + 1, player_pos[1]) not in env.walls
    ):
        player_pos = (player_pos[0] + 1, player_pos[1])
        env.state = player_pos
    elif (
        action == 0
        and player_pos[1] > 0
        and (player_pos[0], player_pos[1] - 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] - 1)
        env.state = player_pos
    elif (
        action == 2
        and player_pos[1] < NUMBER_OF_TILES - 1
        and (player_pos[0], player_pos[1] + 1) not in env.walls
    ):
        player_pos = (player_pos[0], player_pos[1] + 1)
        env.state = player_pos

    x.join()

    # Control the frame rate of the game
    clock.tick(60)
    # Add algorithm name to the display
    font = pygame.font.Font(None, 36)
    text = font.render(f"Algorithm: {current_algorithm}", True, (255, 255, 255))
    screen.blit(text, (10, SCREEN_HEIGHT - 40))
    pygame.display.flip()

# Quit Pygame when the game loop is exited
pygame.quit()