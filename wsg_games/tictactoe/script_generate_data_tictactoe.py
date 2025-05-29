#!/usr/bin/env python3

# python3 wsg_games/tictactoe/script_generate_data_tictactoe.py

import torch as t
import pickle
import os
from wsg_games.tictactoe.analysis.analyse_data import print_data_statistics
from wsg_games.tictactoe.data import (
    cache_tictactoe_data_random,
    train_test_split_tictactoe_first_two_moves_no_overlap,
    create_hard_label_tictactoe_data,
)


def compute_and_save(data_folder, seed: int, device: t.device) -> None:
    print("---" * 20)
    print(f"Data folder: {data_folder}, Seed: {seed}, Device: {device}")
    print("---" * 20)

    # Data
    tictactoe_data = cache_tictactoe_data_random(
        data_folder
        # + "tictactoe_data_200000_no_diagonal_first_two_moves_no_overlap.pkl",
        + "tictactoe_data_first_stone_random_no_diagonal_1000000.pkl",
        device,
        seed=seed,
    )
    print_data_statistics(tictactoe_data)

    # Splits
    (
        tictactoe_train_data,
        tictactoe_weak_finetune_data,
        tictactoe_val_data,
        tictactoe_test_data,
    ) = train_test_split_tictactoe_first_two_moves_no_overlap(
        tictactoe_data, 42, 15, 5, 10, seed
    )
    print("Shape Train: ", tictactoe_train_data.games_data.shape)
    print("Shape Weak Finetune: ", tictactoe_weak_finetune_data.games_data.shape)
    print("Shape Validation: ", tictactoe_val_data.games_data.shape)
    print("Shape Test:  ", tictactoe_test_data.games_data.shape)

    # Hard labels
    tictactoe_train_data = create_hard_label_tictactoe_data(
        tictactoe_train_data, num_samples=1, seed=seed
    )
    tictactoe_weak_finetune_data = create_hard_label_tictactoe_data(
        tictactoe_weak_finetune_data, num_samples=1, seed=seed + 1
    )
    tictactoe_val_data = create_hard_label_tictactoe_data(
        tictactoe_val_data, num_samples=1, seed=seed + 2
    )

    print("Shape Train: ", tictactoe_train_data.games_data.shape)
    tictactoe_train_data.weak_goals_labels[0][:3]
    print("Shape Weak Finetune: ", tictactoe_weak_finetune_data.games_data.shape)
    tictactoe_weak_finetune_data.weak_goals_labels[0][:3]
    print("Shape Val: ", tictactoe_val_data.games_data.shape)
    tictactoe_val_data.weak_goals_labels[0][:3]
    print("Shape Test:  ", tictactoe_test_data.games_data.shape)
    tictactoe_test_data.weak_goals_labels[0][:3]

    all_data_splits = (
        tictactoe_train_data,
        tictactoe_weak_finetune_data,
        tictactoe_val_data,
        tictactoe_test_data,
    )
    all_splits_data_path = os.path.join(data_folder, "tictactoe_all_splits.pkl")

    # Save the tuple
    with open(all_splits_data_path, "wb") as f:
        pickle.dump(all_data_splits, f)
    print(
        f"Saved all data splits (train, weak_finetune, val, test) as a tuple to {all_splits_data_path}"
    )


if __name__ == "__main__":
    # for seed in range(10):
    for seed in [4]:
        print("seed: ", seed)
        data_folder = "/homes/55/bwilop/wsg/data/tictactoe"
        data_folder_seed = data_folder + f"/seed_{seed}/"
        os.makedirs(data_folder_seed, exist_ok=True)
        compute_and_save(
            data_folder_seed,
            seed=seed,
            device=t.device("cpu"),
        )
