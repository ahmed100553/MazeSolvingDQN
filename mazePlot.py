import matplotlib.pyplot as plt
import numpy as np

# Define the 3x3 maze layout
maze_layout = np.array([
    [1, 1, 1],
    [0, 2, 3],
    [1, 1, 1]
])

# Color map: 1 = wall, 0 = path, 2 = player start position, 3 = goal
color_map = {
    1: "gray",     # Wall
    0: "white",    # Path
    2: "blue",     # Start position (Player)
    3: "green"     # Goal position
}

# Plot the maze
plt.figure(figsize=(4, 4))
for i in range(3):
    for j in range(3):
        color = color_map[maze_layout[i, j]]
        plt.gca().add_patch(plt.Rectangle((j, 2 - i), 1, 1, color=color))

# Add grid lines and labels
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.gca().set_xticks(np.arange(0, 3, 1))
plt.gca().set_yticks(np.arange(0, 3, 1))
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.grid(color="black", linewidth=1)
plt.title("3x3 Grid Maze Layout")
plt.gca().invert_yaxis()  # Invert y-axis to match matrix layout
plt.savefig("figures/3x3_maze_layout.png", dpi=300)
plt.show()
