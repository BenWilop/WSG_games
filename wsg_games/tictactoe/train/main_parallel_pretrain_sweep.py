#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_parallel_pretrain_sweep.py

import torch as t
import os

from wsg_games.tictactoe.train.train import run_full_training
from wsg_games.tictactoe.train.save_load_models import (
    save_model,
    load_model_get_matching_files,
)
from wsg_games.tictactoe.data import TicTacToeData, load_split_data
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_training_cfg,
    get_model_config,
)
from wsg_games.tictactoe.data import (
    move_tictactoe_data_to_device,
)
from wsg_games.parallel_computing.parallel_gpu_executor import (
    ParallelGpuExecutor,
    Job,
)


def run_tictactoe_training_task(
    device: t.device,
    data_folder: str,
    goal: Goal,
    model_size: str,
    project_name: str,
    experiment_folder: str,
    training_cfg: dict,
    index: int,
) -> None:
    """Function to be executed by worker process."""
    # skip if already done
    if load_model_get_matching_files(
        project_name, model_size, goal, experiment_folder, index
    ):
        print(
            f"[Task on GPU {device}] already pretrained {goal} and {model_size}, skipping"
        )
        return

    # Load data
    print(f"[Task on GPU {device}] loading data for {goal} and {model_size}")
    tictactoe_train_data, _, tictactoe_val_data, tictactoe_test_data = load_split_data(
        data_folder, device, index
    )

    # Train
    print(f"[Task on GPU {device}] starting training for {goal} and {model_size}")
    model_cfg = get_model_config(model_size)
    model_cfg.device = device
    model, experiment_name, run_id = run_full_training(
        project_name,
        model_size,
        goal,
        tictactoe_train_data,
        tictactoe_val_data,
        tictactoe_test_data,
        training_cfg,
        model_cfg,
    )

    # Save
    save_model(model, run_id, project_name, experiment_name, experiment_folder, index)
    print(f"[Task on GPU {device}] finished {goal} and {model_size} (run {run_id})")


def main_pretrain(
    data_folder: str, experiment_folder: str, project_name: str, n_indices: int
):
    print("Starting pretraining...")
    print(f"  data_folder: {data_folder}")
    print(f"  project_name: {project_name}")
    print(f"  experiment_folder: {experiment_folder}")
    print(f"  n_indices: {n_indices}")

    project_experiment_folder = os.path.join(experiment_folder, project_name)
    os.makedirs(project_experiment_folder, exist_ok=True)

    training_cfg = get_training_cfg()

    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    goals = [Goal.WEAK_GOAL, Goal.STRONG_GOAL]

    all_jobs: list[Job] = []
    for index in range(n_indices):
        print(f"\n--- Preparing jobs for Index {index} ---")
        jobs_for_index = 0
        for goal in goals:
            for model_size in model_sizes:
                job_args = (
                    data_folder,
                    goal,
                    model_size,
                    project_name,
                    experiment_folder,
                    training_cfg,
                    index,
                )
                all_jobs.append(
                    Job(function=run_tictactoe_training_task, args=job_args)
                )
                jobs_for_index += 1
        print(
            f"  Prepared {jobs_for_index} jobs for Index {index}. Total jobs scheduled so far: {len(all_jobs)}"
        )

    parallel_gpu_executor = ParallelGpuExecutor()
    parallel_gpu_executor.submit_jobs(all_jobs)
    print("\nPretraining script finished.")


if __name__ == "__main__":
    data_folder = "/homes/55/bwilop/wsg/data/tictactoe"
    experiment_folder = "/homes/55/bwilop/wsg/experiments/tictactoe"
    project_name = "tictactoe_pretraining"
    n_indices = 10
    main_pretrain(data_folder, experiment_folder, project_name, n_indices)
