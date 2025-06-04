#!/usr/bin/env python3

# python3 wsg_games/tictactoe/crosscoder/script_compute_activations_pretrain.py

import torch as t

from wsg_games.tictactoe.data import load_split_data
from wsg_games.tictactoe.game import Goal
from wsg_games.parallel_computing.parallel_gpu_executor import (
    ParallelGpuExecutor,
    Job,
)
from wsg_games.meta import *
from wsg_games.tictactoe.crosscoder.collect_activations import compute_activations


def run_compute_activations_pretrain_task(
    device: t.device,
    data_folder: str,
    experiment_folder: str,
    project_name_pretrain: str,
    crosscoder_folder: str,
    game: Game,
    goal: Goal,
    model_size: str,
    index: int,
) -> None:
    """Function to be executed by worker process."""
    # Load data
    print(f"[Task on GPU {device}] loading data for {goal} and {model_size}")
    _, _, tictactoe_val_data, tictactoe_test_data = load_split_data(
        data_folder, device, index
    )

    # Run
    # Saves automatically and skips if already exists
    print(
        f"[Task on GPU {device}] starting computation of activations for {goal} and {model_size}"
    )
    compute_activations(
        game,
        goal,
        project_name_pretrain,
        None,
        None,
        model_size,
        index,
        crosscoder_folder,
        tictactoe_test_data.games_data,
        tictactoe_val_data.games_data,
        experiment_folder,
        device=device,
    )

    print(f"[Task on GPU {device}] finished {index} for {goal} and {model_size}.")


def main_compute_activations_pretrain(
    data_folder: str,
    experiment_folder: str,
    project_name_pretrain: str,
    crosscoder_folder: str,
    game: Game,
    n_indices: int,
):
    print("Starting pretraining...")
    print(f"  data_folder: {data_folder}")
    print(f"  experiment_folder: {experiment_folder}")
    print(f"  n_indices: {n_indices}")

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
                    experiment_folder,
                    project_name_pretrain,
                    crosscoder_folder,
                    game,
                    goal,
                    model_size,
                    index,
                )
                all_jobs.append(
                    Job(function=run_compute_activations_pretrain_task, args=job_args)
                )
                jobs_for_index += 1
        print(
            f"  Prepared {jobs_for_index} jobs for Index {index}. Total jobs scheduled so far: {len(all_jobs)}"
        )

    parallel_gpu_executor = ParallelGpuExecutor(ngpus=1)
    parallel_gpu_executor.submit_jobs(all_jobs)
    print("\Computing activations pretrain script finished.")


if __name__ == "__main__":
    data_folder = "/homes/55/bwilop/wsg/data/tictactoe/"
    experiment_folder = "/homes/55/bwilop/wsg/experiments/tictactoe/"
    project_name_pretrain = "tictactoe_pretraining6"
    crosscoder_folder = experiment_folder + "tictactoe/crosscoder6/"
    game = Game.TICTACTOE
    n_indices = 10
    main_compute_activations_pretrain(
        data_folder,
        experiment_folder,
        project_name_pretrain,
        crosscoder_folder,
        game,
        n_indices,
    )
