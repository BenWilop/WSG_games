import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from wsg_games.tictactoe.data import TicTacToeData


def visualize_game(data: TicTacToeData, game_id: int, model):
    """
    Each row is one move of the game, marked as 'x'. Each plot is soft labels.
    Column 0: Model output
    Column 1: Random label
    Column 2: Weak goal label
    """
    game_moves = data.games_data[game_id]
    print(game_moves)

    # Build board
    board_states = []
    board = [""] * 9
    current_player = "X"
    for move in game_moves:
        if move < 9:
            board[move] = current_player
            current_player = "O" if current_player == "X" else "X"
        board_states.append(board.copy())
    n_moves = len(board_states)

    model_labels = softmax(model(data.games_data[game_id]), dim=-1)

    # Plot
    fig, axes = plt.subplots(n_moves, 4, figsize=(16, 4 * n_moves))
    if n_moves == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_moves - 1):
        # Get data
        current_state = board_states[i]

        # Get labels
        model_label = model_labels[:, i, :]
        random_label = data.random_move_labels[game_id, i, :]
        weak_label = data.weak_goals_labels[game_id, i, :]
        strong_label = data.strong_goals_labels[game_id, i, :]
        distributions = [model_label, random_label, weak_label, strong_label]
        titles = [
            "Model Output",
            "Random Label",
            "Weak Goal Label",
            "Strong Goal Label",
        ]

        # Plot
        for j, (dist, title_prefix) in enumerate(zip(distributions, titles)):
            ax = axes[i, j]
            dist = dist.detach().cpu().numpy().flatten()
            board_grid = dist[:9].reshape(3, 3)
            end_game_prob = dist[9]

            # Color map
            im = ax.imshow(board_grid, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"{title_prefix} (End-of-game: {end_game_prob:.2f})")
            ax.set_xticks([])
            ax.set_yticks([])

            # Write 'X' and 'O'
            for pos, symbol in enumerate(current_state):
                if symbol:  # if the cell is occupied
                    row, col = divmod(pos, 3)
                    ax.text(
                        col,
                        row,
                        symbol,
                        ha="center",
                        va="center",
                        fontsize=16,
                        color="white",
                    )

    # Color legend
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    plt.tight_layout(rect=[0, 0, 0.93, 1])

    plt.show()
