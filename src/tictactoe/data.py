import random
import math
from typing import Optional, Callable
from enum import Enum
from copy import deepcopy
import numpy as np
import torch as t
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm

from tictactoe.game import *


class TicTacToeData(Enum):
    GAMES_DATA: Float[Tensor, "n_games game_length"]  # All games get filled up to length 10: [10, move1, ... movek, potentially 9's] (last element removed as we do next move prediction)
    RANDOM_MOVE_LABELS: Float[Tensor, "n_games game_length n_tokens"]  # 1/k on all k legal moves, 0 for other tokens
    WEAK_GOAL_LABELS: Float[Tensor, "n_games game_length n_tokens"]  # 1/k on all k optimal moves under weak goal, 0 for other tokens
    STRONG_GOAL_LABELS: Float[Tensor, "n_games game_length n_tokens"]


def _next_possible_moves(seq: list[int]) -> list[int]:
    board = Board()
    for move in seq[1:]:
        try:
            board.make_move(move)
        except:
            return [9]
    if board.game_state != State.ONGOING:
        return [9]
    else:
        return board.get_possible_moves()

def _next_minimax_moves(seq: list[int], goal: Goal) -> list[int]:
    board = Board()
    for move in seq[1:]:
        try:
            board.make_move(move)
        except:
            return [9]
    if board.game_state != State.ONGOING:
        return [9]
    else:
        return get_best_moves(board, goal)
    
def _get_label_tensor(seq, cache, compute_moves_fn):
    seq_key: str = str(seq)
    if seq_key not in cache:
        cache[seq_key] = compute_moves_fn(seq)
    moves = cache[seq_key]
    label_vector = [1.0 if i in moves else 0.0 for i in range(9)]
    label_tensor = t.tensor(label_vector, requires_grad=False)
    return label_tensor / t.sum(label_tensor)

def calculate_tictactoe_data() -> TicTacToeData:
    games = generate_all_games([Board()])
    games_tensor = t.tensor(
        [
            [10] + game.moves_played + ([9] * (10 - len(game.moves_played)))
            for game in games
        ],
        requires_grad=False,
    )
    games_data = games_tensor[:, :-1]
    random_labels_all: list[Float[Tensor, "game_length n_tokens"]] = []
    weak_labels_all = []
    strong_labels_all = []
    cache_random_label: dict["str", list[int]] = {}  # game sequence -> result of _next_possible_moves
    cache_weak_label = {}
    cache_strong_label = {}
    for game in tqdm(games_data):
        random_label: list[Float[Tensor, "n_tokens"]] = []
        weak_label = []
        strong_label = []
        for idx in range(len(game)):
            seq: list[int] = game[:idx+1].tolist()
            random_label.append(
                _get_label_tensor(seq, cache_random_label, _next_possible_moves)
            )
            weak_label.append(
                _get_label_tensor(seq, cache_weak_label, lambda s: _next_minimax_moves(s, Goal.WEAK_GOAL))
            )
            strong_label.append(
                _get_label_tensor(seq, cache_strong_label, lambda s: _next_minimax_moves(s, Goal.STRONG_GOAL))
            )
        random_labels_all.append(t.stack(random_label))
        weak_labels_all.append(t.stack(weak_label))
        strong_labels_all.append(t.stack(strong_label))
    
    return TicTacToeData(
        GAMES_DATA=games_data,
        RANDOM_MOVE_LABELS=t.stack(random_labels_all),
        WEAK_GOAL_LABELS=t.stack(weak_labels_all),
        STRONG_GOAL_LABELS=t.stack(strong_labels_all),
    )
