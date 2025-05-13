#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_random_models.py

import torch as t
import os

from wsg_games.tictactoe.train.save_load_models import (
    save_model,
    load_model_get_matching_files,
)
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_model_config,
)
from datetime import datetime
from transformer_lens import HookedTransformer
from transformer_lens.utilities.devices import move_to_and_update_config


def main_pretrain_random(
    data_folder: str, experiment_folder: str, project_name: str, n_indices: int
):
    print("Starting pretraining...")
    print(f"  data_folder: {data_folder}")
    print(f"  project_name: {project_name}")
    print(f"  experiment_folder: {experiment_folder}")
    print(f"  n_indices: {n_indices}")

    project_experiment_folder = os.path.join(experiment_folder, project_name)
    os.makedirs(project_experiment_folder, exist_ok=True)

    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    goals = [Goal.WEAK_GOAL, Goal.STRONG_GOAL]
    for index in range(n_indices):
        for goal in goals:
            for model_size in model_sizes:
                if load_model_get_matching_files(
                    project_name, model_size, goal, experiment_folder, index
                ):
                    print(
                        f"Random model index {index}, {goal} and {model_size} already exists, skipping"
                    )
                    return

                model_cfg = get_model_config(model_size)
                model = HookedTransformer(model_cfg)
                model = move_to_and_update_config(model, model_cfg.device)

                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
                experiment_name = f"{model_size}_{str(goal)}_{timestamp}"

                save_model(
                    model,
                    "RANDOM",
                    project_name,
                    experiment_name,
                    experiment_folder,
                    index,
                )
                print(f"Random model index {index}, {goal} and {model_size} finished.")

    print("\nFinished creating random models")


if __name__ == "__main__":
    data_folder = "/homes/55/bwilop/wsg/data/tictactoe"
    experiment_folder = "/homes/55/bwilop/wsg/experiments/tictactoe"
    project_name = "tictactoe_pretraining_random"
    n_indices = 10
    main_pretrain_random(data_folder, experiment_folder, project_name, n_indices)
