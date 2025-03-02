import math
from typing import Optional
from enum import Enum
from copy import deepcopy
import torch as t
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F

from wsg_games.tictactoe.data import TicTacToeData

def evaluate_logits(logits: Float[Tensor, "n_games game_length n_tokens"], tictactoe_data: TicTacToeData, loss_fn) -> dict[str, float]:
    """Returns dictionary metric -> value"""
    res = {}
    res['weak_loss'] = loss_fn(logits, tictactoe_data.weak_goals_labels).item()
    res['strong_loss'] = loss_fn(logits, tictactoe_data.strong_goals_labels).item()
    
    predictions = F.softmax(logits, dim=1)

    # Pprediction is correct if the max token has a positive label.
    pred_indices = t.argmax(predictions, dim=-1)  # shape: [n_games, game_length]
    n_games, game_length, _ = predictions.shape
    game_indices = t.arange(n_games).unsqueeze(1).expand(n_games, game_length)
    move_indices = t.arange(game_length).unsqueeze(0).expand(n_games, game_length)
    weak_correct = (tictactoe_data.weak_goals_labels[game_indices, move_indices, pred_indices] > 0)
    res['weak_accuracy'] = weak_correct.float().mean().item()
    strong_correct = (tictactoe_data.strong_goals_labels[game_indices, move_indices, pred_indices] > 0)
    res['strong_accuracy'] = strong_correct.float().mean().item()

    # Move is legal if the random move label is positive.
    illegal_mask = (tictactoe_data.random_move_labels == 0).float()  # shape: [n_games, game_length, n_tokens]
    illegal_move_chance = (predictions * illegal_mask).sum(dim=-1).mean().item()
    res['illegal_move_chance'] = illegal_move_chance

    return res