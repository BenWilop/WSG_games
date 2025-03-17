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
        except ValueError:
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
        except ValueError:
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

def label_games_tensor(games_tensor: t.Tensor) -> TicTacToeData:
    games_data = games_tensor[:, :-1]
    random_labels_all = []
    weak_labels_all = []
    strong_labels_all = []
    cache_random_label = {}
    cache_weak_label = {}
    cache_strong_label = {}
    for game in tqdm(games_data, desc="Labeling games"):
        random_label = []
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


def calculate_tictactoe_data() -> TicTacToeData:
    games = generate_all_games([Board()])
    games_tensor = t.tensor(
        [
            [10] + game.moves_played + ([9] * (10 - len(game.moves_played)))
            for game in games
        ],
        requires_grad=False,
    )
    return label_games_tensor(games_tensor)


def calculate_tictactoe_data_random(n_samples: int) -> TicTacToeData:
    games = []
    for _ in tqdm(range(n_samples), desc="Generating random games"):
        board = Board()
        while board.game_state == State.ONGOING:
            legal_moves = board.get_possible_moves()
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            board.make_move(move)
        games.append(board)
    games_tensor = t.tensor(
        [
            [10] + game.moves_played + ([9] * (10 - len(game.moves_played)))
            for game in games
        ],
        requires_grad=False,
    )
    return label_games_tensor(games_tensor)


def split_data_by_indices(tictactoe_data, train_inds, weak_finetune_inds, val_inds, test_inds, device=None):
    def to_device(data):
        return data.to(device) if device else data

    def build_dataset(indices):
        return TicTacToeData(
            games_data=to_device(tictactoe_data.games_data[indices]),
            random_move_labels=to_device(tictactoe_data.random_move_labels[indices]),
            weak_goals_labels=to_device(tictactoe_data.weak_goals_labels[indices]),
            strong_goals_labels=to_device(tictactoe_data.strong_goals_labels[indices]),
        )

    train_data = build_dataset(train_inds)
    val_data = build_dataset(val_inds)
    test_data = build_dataset(test_inds)
    weak_finetune_data = build_dataset(weak_finetune_inds)

    return train_data, weak_finetune_data, val_data, test_data


def train_test_split_tictactoe_first(tictactoe_data, train_ratio, weak_finetune_ratio, val_ratio, test_ratio, device=None, seed=None):
    if abs(train_ratio + weak_finetune_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/Val/Test ratios must sum to 1.")

    num_games = len(tictactoe_data.games_data)

    inds = t.randperm(num_games, generator=t.Generator().manual_seed(seed))

    n_train = int(train_ratio * num_games)
    n_finetune = int(weak_finetune_ratio * num_games)
    n_val = int(val_ratio * num_games)

    n_1 = n_train
    n_2 = n_1 + n_finetune
    n_3 = n_2 + n_val
    train_inds = inds[:n_1]
    weak_finetune_inds = inds[n_1:n_2]
    val_inds = inds[n_2:n_3]
    test_inds = inds[n_3:]

    return split_data_by_indices(tictactoe_data, train_inds, weak_finetune_inds, val_inds, test_inds)


def train_test_split_tictactoe_first_two_moves_no_overlap(tictactoe_data, n_first_two_train, n_first_two_weak_finetune, n_first_two_val, n_first_two_test, device=None, seed=None):
    if n_first_two_train + n_first_two_weak_finetune + n_first_two_val + n_first_two_test != 72:
        raise ValueError("The sum of first-two-move splits must equal 72.")

    unique_first_two_moves = t.unique(tictactoe_data.games_data[:, :3], dim=0)
    shuffled_indices = t.randperm(len(unique_first_two_moves), generator=t.Generator().manual_seed(seed))

    n_1 = n_first_two_train
    n_2 = n_1 + n_first_two_weak_finetune
    n_3 = n_2 + n_first_two_val
    train_first_two_moves = unique_first_two_moves[shuffled_indices[:n_1]]
    weak_finetune_first_two_moves = unique_first_two_moves[shuffled_indices[n_1:n_2]]
    val_first_two_moves = unique_first_two_moves[shuffled_indices[n_2:n_3]]
    test_first_two_moves = unique_first_two_moves[shuffled_indices[n_3:]]

    def indices_for_first_two_moves(first_two_moves):
        mask = (tictactoe_data.games_data[:, None, :3] == first_two_moves[None, :, :]).all(dim=2).any(dim=1)
        return mask.nonzero().flatten()

    train_inds = indices_for_first_two_moves(train_first_two_moves)
    weak_finetune_inds = indices_for_first_two_moves(weak_finetune_first_two_moves)
    val_inds = indices_for_first_two_moves(val_first_two_moves)
    test_inds = indices_for_first_two_moves(test_first_two_moves)

    return split_data_by_indices(tictactoe_data, train_inds, weak_finetune_inds, val_inds, test_inds, device)


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

def cache_tictactoe_data_(path: str) -> TicTacToeData:
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

def cache_tictactoe_data_random(path: str) -> TicTacToeData:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, TicTacToeData), f"Data loaded from {path} is not a TicTacToeData object"
        return data
    else:
        data = calculate_tictactoe_data_random(200000)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return data