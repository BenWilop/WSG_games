#!/usr/bin/env python3

# python3 wsg_games/tictactoe/crosscoder/script_compute_activations_finetune.py

import torch as t

from wsg_games.tictactoe.data import load_split_data
from wsg_games.tictactoe.game import Goal
from wsg_games.parallel_computing.parallel_gpu_executor import (
    ParallelGpuExecutor,
    Job,
)
from wsg_games.meta import *
from wsg_games.tictactoe.crosscoder.collect_activations import compute_activations


def run_compute_activations_finetune_task(
    device: t.device,
    data_folder: str,
    experiment_folder: str,
    project_name_finetune: str,
    crosscoder_folder: str,
    game: Game,
    weak_model_size: str,
    strong_model_size: str,
    index: int,
) -> None:
    """Function to be executed by worker process."""
    # Load data
    print(
        f"[Task on GPU {device}] loading data for {weak_model_size} and {strong_model_size}"
    )
    _, _, tictactoe_val_data, tictactoe_test_data = load_split_data(
        data_folder, device, index
    )

    # Run
    # Saves automatically and skips if already exists
    print(
        f"[Task on GPU {device}] starting computation of activations for {weak_model_size} and {strong_model_size}"
    )
    compute_activations(
        game,
        None,
        None,
        weak_model_size,
        project_name_finetune,
        strong_model_size,
        index,
        crosscoder_folder,
        tictactoe_test_data.games_data,
        tictactoe_val_data.games_data,
        experiment_folder,
        device=device,
    )

    print(
        f"[Task on GPU {device}] finished {index} for {weak_model_size} and {strong_model_size}."
    )


def main_compute_activations_finetune(
    data_folder: str,
    experiment_folder: str,
    project_name_finetune: str,
    crosscoder_folder: str,
    game: Game,
    indices: list[int],
):
    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    finetuning_pairs = [
        (weak_size, model_sizes[j])
        for i, weak_size in enumerate(model_sizes)
        for j in range(i + 1, len(model_sizes))  # strong_size > weak_size index
    ]

    all_jobs: list[Job] = []
    for index in indices:
        print(f"\n--- Preparing jobs for Index {index} ---")
        jobs_for_index = 0
        for weak_model_size, strong_model_size in finetuning_pairs:
            job_args = (
                data_folder,
                experiment_folder,
                project_name_finetune,
                crosscoder_folder,
                game,
                weak_model_size,
                strong_model_size,
                index,
            )
            all_jobs.append(
                Job(function=run_compute_activations_finetune_task, args=job_args)
            )
            jobs_for_index += 1
        print(
            f"  Prepared {jobs_for_index} jobs for Index {index}. Total jobs scheduled so far: {len(all_jobs)}"
        )

    parallel_gpu_executor = ParallelGpuExecutor(ngpus=1)
    parallel_gpu_executor.submit_jobs(all_jobs)
    print("\Computing activations finetune script finished.")


if __name__ == "__main__":
    data_folder = "/homes/55/bwilop/wsg/data/tictactoe/"
    experiment_folder = "/homes/55/bwilop/wsg/experiments/tictactoe/"
    project_name_finetune = "tictactoe_finetuning_use_best_val_step6_lre5"
    crosscoder_folder = experiment_folder + "tictactoe/crosscoder/"
    game = Game.TICTACTOE
    indices = [5, 6, 7, 8, 9]
    main_compute_activations_finetune(
        data_folder,
        experiment_folder,
        project_name_finetune,
        crosscoder_folder,
        game,
        indices,
    )
