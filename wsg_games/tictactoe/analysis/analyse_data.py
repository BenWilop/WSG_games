import torch.distributions as dist
import torch as t

from wsg_games.tictactoe.data import TicTacToeData

def entropy(labels: t.Tensor) -> float:
    """Minimal achievable CE loss"""
    distribution = dist.Categorical(probs=labels)
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
    
