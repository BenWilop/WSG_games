import os
import math
import pickle
from copy import deepcopy
import numpy as np
import torch as t
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm
from dataclasses import dataclass

from wsg_games.tictactoe.game import *


@dataclass
class TicTacToeData:
    games_data: Float[Tensor, "n_games game_length"]  # All games get filled up to length 10: [10, move1, ... movek, potentially 9's] (last element removed as we do next move prediction)
    random_move_labels: Float[Tensor, "n_games game_length n_tokens"]  # 1/k on all k legal moves, 0 for other tokens
    weak_goals_labels: Float[Tensor, "n_games game_length n_tokens"]  # 1/k on all k optimal moves under weak goal, 0 for other tokens
    strong_goals_labels: Float[Tensor, "n_games game_length n_tokens"]


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
    assert len(moves) > 0
    label_vector = [1.0 if i in moves else 0.0 for i in range(10)]  # can predict 9 as well to stop game
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
        games_data=games_data,
        random_move_labels=t.stack(random_labels_all),
        weak_goals_labels=t.stack(weak_labels_all),
        strong_goals_labels=t.stack(strong_labels_all),
    )

def train_test_split_tictactoe(
    tictactoe_data: TicTacToeData,
    split_ratio: float = 0.8,
    device: str | None = None,
    seed: str | None = None,
):
    if seed is not None:
        t.random.manual_seed(seed)

    # Check that data is coeherent
    assert 0 <= split_ratio <= 1
    n_games = tictactoe_data.games_data.shape[0]
    assert n_games == len(tictactoe_data.random_move_labels)
    assert n_games == len(tictactoe_data.weak_goals_labels)
    assert n_games == len(tictactoe_data.strong_goals_labels)

    # Indices
    inds = t.randperm(n_games)
    split = math.floor(split_ratio * n_games)
    train_inds, test_inds = inds[:split], inds[split:]

    # Data
    games_train = tictactoe_data.games_data[train_inds]
    random_train = tictactoe_data.random_move_labels[train_inds]
    weak_train = tictactoe_data.weak_goals_labels[train_inds]
    strong_train = tictactoe_data.strong_goals_labels[train_inds]
    games_test = tictactoe_data.games_data[test_inds]
    random_test = tictactoe_data.random_move_labels[test_inds]
    weak_test = tictactoe_data.weak_goals_labels[test_inds]
    strong_test = tictactoe_data.strong_goals_labels[test_inds]

    if device is not None:
        games_train = games_train.to(device)
        random_train = random_train.to(device)
        weak_train = weak_train.to(device)
        strong_train = strong_train.to(device)
        games_test = games_test.to(device)
        random_test = random_test.to(device)
        weak_test = weak_test.to(device)
        strong_test = strong_test.to(device)

    train_data = TicTacToeData(
        games_data=games_train,
        random_move_labels=random_train,
        weak_goals_labels=weak_train,
        strong_goals_labels=strong_train,
    )
    test_data = TicTacToeData(
        games_data=games_test,
        random_move_labels=random_test,
        weak_goals_labels=weak_test,
        strong_goals_labels=strong_test,
    )
    return train_data, test_data

def random_sample_tictactoe_data(tictactoe_data: TicTacToeData, n_samples: int) -> TicTacToeData:
    n_games = len(tictactoe_data.games_data)
    assert 0 < n_samples <= n_games
    sample_inds = t.randperm(n_games)[:n_samples]
    sampled_data = TicTacToeData(
        games_data = tictactoe_data.games_data[sample_inds],
        random_move_labels = tictactoe_data.random_move_labels[sample_inds],
        weak_goals_labels = tictactoe_data.weak_goals_labels[sample_inds],
        strong_goals_labels = tictactoe_data.strong_goals_labels[sample_inds]
    )
    return sampled_data

def cache_tictactoe_data(path: str) -> TicTacToeData:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, TicTacToeData), f"Data loaded from {path} is not a TicTacToeData object"
        return data
    else:
        data = calculate_tictactoe_data()
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return data
