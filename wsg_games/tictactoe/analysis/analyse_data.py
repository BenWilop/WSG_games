import torch.distributions as dist
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass


from wsg_games.tictactoe.evals import evaluate_predictions
from wsg_games.tictactoe.data import TicTacToeData


def entropy(labels: t.Tensor) -> float:
    """Minimal achievable CE loss"""
    sliced_labels = labels[:, 3:, :]
    distribution = dist.Categorical(probs=sliced_labels)
    ent = distribution.entropy()
    return ent.mean().item()


def print_data_statistics(tictactoe_data: TicTacToeData) -> None:
    print("Shape Data:   ", tictactoe_data.games_data.shape)
    print("Shape Random: ", tictactoe_data.random_move_labels.shape)
    print("Shape Weak:   ", tictactoe_data.weak_goals_labels.shape)
    print("Shape Strong: ", tictactoe_data.strong_goals_labels.shape)
    print("--------------------------------------------------------")
    print("Evals Random: ", evaluate_predictions(tictactoe_data.random_move_labels, tictactoe_data))
    print("Evals Weak:   ", evaluate_predictions(tictactoe_data.weak_goals_labels, tictactoe_data))
    print("Evals Strong: ", evaluate_predictions(tictactoe_data.strong_goals_labels, tictactoe_data))
    print("--------------------------------------------------------")
    print("Entropy Random: ", entropy(tictactoe_data.random_move_labels))
    print("Entropy Weak:   ", entropy(tictactoe_data.weak_goals_labels))
    print("Entropy Strong: ", entropy(tictactoe_data.strong_goals_labels))
    print("--------------------------------------------------------")
    

def get_all_prefixes(games_data: t.Tensor) -> set:
    games_data = games_data.cpu()
    prefixes = set()
    n_games, game_length = games_data.shape
    for i in range(n_games):
        game = games_data[i]
        for l in range(1, game_length + 1):
            prefix = tuple(game[:l].tolist())
            prefixes.add(prefix)
    return prefixes


def calculate_leakage_percentage(train_data: TicTacToeData, test_data: TicTacToeData) -> float:
    """
    Computes the percentage of test prefixes (game states) that also appear in the train data.
    """
    train_prefixes = get_all_prefixes(train_data.games_data)
    test_prefixes = get_all_prefixes(test_data.games_data)
    
    leaked = 0
    total = 0
    test_games = test_data.games_data.cpu()
    n_games, game_length = test_games.shape
    for i in range(n_games):
        game = test_games[i]
        for l in range(3, game_length + 1):
            total += 1
            prefix = tuple(game[:l].tolist())
            if prefix in train_prefixes:
                leaked += 1
    leakage_pct = (leaked / total) * 100

    return leakage_pct


def get_bin_index(count: int, bin_edges: list[int]) -> int:
    for i, edge in enumerate(bin_edges):
        if count <= edge:
            return i
    return len(bin_edges) - 1


def compute_prefix_histograms(games_tensor: t.Tensor, bin_edges: list[int]) -> tuple[list, list]:
    prefix_counts = Counter()
    games_np = games_tensor.cpu().numpy()
    n_games, game_length = games_np.shape
    for i in range(n_games):
        game = games_np[i]
        prefix = []
        for l in range(game_length):
            prefix.append(game[l])
            prefix_tuple = tuple(prefix)
            prefix_counts[prefix_tuple] += 1

    unweighted_hist = [0] * len(bin_edges)
    weighted_hist = [0] * len(bin_edges)
    for freq in prefix_counts.values():
        bin_index = get_bin_index(freq, bin_edges)
        unweighted_hist[bin_index] += 1
        weighted_hist[bin_index] += freq
    return unweighted_hist, weighted_hist


def plot_binned_histograms(unweighted_hist: list, weighted_hist: list, title: str, bin_edges: list[str]):
    total_unique = sum(unweighted_hist)
    total_occurrences = sum(weighted_hist)
    unweighted_pct = [100 * count / total_unique for count in unweighted_hist]
    weighted_pct = [100 * count / total_occurrences for count in weighted_hist]

    # Labels
    bin_labels = []
    for i, edge in enumerate(bin_edges):
        bin_labels.append(f"<={edge}")
    
    x = np.arange(len(bin_edges))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, unweighted_pct, width, label='Unweighted', color='blue')
    plt.bar(x + width/2, weighted_pct, width, label='Weighted', color='orange')
    # plt.yscale("log")
    plt.xlabel("Prefix Frequency Bins")
    plt.ylabel("Percentage")
    plt.xticks(x, bin_labels)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_train_test_prefix_histograms(train_data: TicTacToeData, test_data: TicTacToeData, bin_edges: list[str]=[1, 10, 100, 1000, 10000, 100000, 1000000]):
    train_unweighted, train_weighted = compute_prefix_histograms(train_data.games_data, bin_edges)
    test_unweighted, test_weighted = compute_prefix_histograms(test_data.games_data, bin_edges)
    
    plot_binned_histograms(train_unweighted, train_weighted, "Train Data Prefix Frequency Histogram", bin_edges)
    plot_binned_histograms(test_unweighted, test_weighted, "Test Data Prefix Frequency Histogram", bin_edges)
