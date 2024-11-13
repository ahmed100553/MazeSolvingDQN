import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  # Ensure numpy is imported

# Define the maze layout
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

def create_markov_graph(level):
    num_rows = len(level)
    num_cols = len(level[0])
    G = nx.DiGraph()

    # Directions: left, up, right, down
    movements = {
        'left': (0, -1),
        'up': (-1, 0),
        'right': (0, 1),
        'down': (1, 0),
    }

    for r in range(num_rows):
        for c in range(num_cols):
            if level[r][c] != 'X':
                current_state = (r, c)
                G.add_node(current_state)
                for dr, dc in movements.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < num_rows and 0 <= nc < num_cols and level[nr][nc] != 'X':
                        next_state = (nr, nc)
                        G.add_edge(current_state, next_state)

    return G

def plot_markov_graph(G, level, start_pos, goal_pos):
    num_rows = len(level)
    num_cols = len(level[0])
    
    # Create a grid to represent the maze
    maze_array = np.array([[1 if cell == 'X' else 0 for cell in row] for row in level])

    # Positions for nodes
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Adjust for correct orientation

    plt.figure(figsize=(12, 12))

    # Display the maze layout
    plt.imshow(maze_array, cmap='Greys', origin='upper', extent=(-0.5, num_cols - 0.5, num_rows - 0.5, -0.5))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', edgecolors='black')

    # Highlight the start and goal positions
    nx.draw_networkx_nodes(G, pos, nodelist=[start_pos], node_size=150, node_color='green', edgecolors='black', label='Start')
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_pos], node_size=150, node_color='red', edgecolors='black', label='Goal')

    # Draw the edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='blue')

    # Optionally add labels
    # labels = {node: f"{node}" for node in G.nodes()}
    # nx.draw_networkx_labels(G, pos, labels, font_size=6)

    # Add a legend
    plt.legend(scatterpoints=1, markerscale=1, fontsize=12)

    plt.title('Markov Representation of the Maze')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Identify start and goal positions
    start_pos = None
    goal_pos = None
    for r, row in enumerate(level):
        for c, cell in enumerate(row):
            if cell == 'P':
                start_pos = (r, c)
            elif cell == 'G':  # If 'G' represents the goal in the maze layout
                goal_pos = (r, c)
    if not goal_pos:
        goal_pos = (1, 2)  # Use specified goal position if not in the maze layout

    G = create_markov_graph(level)