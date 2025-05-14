#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_parallel_finetune_sweep.py
import queue
import torch as t
import os
import torch.multiprocessing as mp
from multiprocessing import Manager
import json
import copy

from wsg_games.tictactoe.train.finetune import finetune_strong_with_weak
from wsg_games.tictactoe.train.save_load_models import (
    load_model,
    save_model,
    load_finetuned_model_get_matching_files,
)
from transformer_lens.utilities.devices import move_to_and_update_config
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_training_cfg_finetune,
)
from wsg_games.tictactoe.data import (
    move_tictactoe_data_to_device,
    load_split_data,
)
from wsg_games.parallel_computing.parallel_gpu_executor import (
    ParallelGpuExecutor,
    Job,
)


def run_tictactoe_finetuning_task(
    device: t.device,
    data_folder: str,
    experiment_folder: str,
    pretrained_project_name_weak: str,
    pretrained_project_name_strong: str,
    finetuned_project_name: str,
    weak_size: str,
    strong_size: str,
    training_cfg_finetune: dict,
    index: int,
) -> None:
    """Function to be executed by worker process."""
    task_id = f"Index {index}, Weak {weak_size} -> Strong {strong_size}"
    print(f"[Task {task_id} on GPU {device}] preparing on device {device}")

    # skip if already done
    if load_finetuned_model_get_matching_files(
        finetuned_project_name, weak_size, strong_size, experiment_folder, index
    ):
        print(f"[Task {task_id} on GPU {device}] already finetuned, skipping.")
        return

    # Load data
    print(f"[Task {task_id} on GPU {device}] loading data.")
    _, tictactoe_weak_finetune_data, tictactoe_val_data, tictactoe_test_data = (
        load_split_data(data_folder, index)
    )

    # Move data to device
    print(f"[Task {task_id} on GPU {device}] preparing.")
    tictactoe_weak_finetune_data = move_tictactoe_data_to_device(
        tictactoe_weak_finetune_data, device=device
    )
    tictactoe_val_data = move_tictactoe_data_to_device(
        tictactoe_val_data, device=device
    )
    tictactoe_test_data = move_tictactoe_data_to_device(
        tictactoe_test_data, device=device
    )

    # load pretrained weak & strong
    weak_model = load_model(
        pretrained_project_name_weak,
        weak_size,
        Goal.WEAK_GOAL,
        experiment_folder,
        device,
        index,
    )
    strong_model = load_model(
        pretrained_project_name_strong,
        strong_size,
        Goal.STRONG_GOAL,
        experiment_folder,
        device,
        index,
    )

    if weak_model is None or strong_model is None:
        print(
            f"[Task {task_id} on GPU {device}] Missing pretrained model for weak={weak_size} or strong={strong_size} (Index {index}). Skipping."
        )
        return

    weak_model = move_to_and_update_config(weak_model, device)
    strong_model = move_to_and_update_config(strong_model, device)

    # Finetune
    finetuned_model, experiment_name, run_id = finetune_strong_with_weak(
        finetuned_project_name,
        weak_model,
        weak_size,
        strong_model,
        strong_size,
        tictactoe_weak_finetune_data,
        tictactoe_val_data,
        tictactoe_test_data,
        training_cfg_finetune,
    )

    # save
    save_model(
        finetuned_model,
        run_id,
        finetuned_project_name,
        experiment_name,
        experiment_folder,
        index,
    )
    print(f"[Worker {device}] finished {weak_size}→{strong_size} (run {run_id})")
    t.cuda.empty_cache()


def main_finetune_multi_index(
    data_folder: str,
    experiment_folder: str,
    pretrained_project_name_weak: str,
    pretrained_project_name_strong: str,
    finetuned_project_name: str,
    n_indices: int,
    training_cfg_finetune: dict,
):
    print("Starting finetuning...")
    print(f"  data_folder: {data_folder}")
    print(f"  experiment_folder: {experiment_folder}")
    print(f"  pretrained_project_name_weak: {pretrained_project_name_weak}")
    print(f"  pretrained_project_name_strong: {pretrained_project_name_strong}")
    print(f"  finetuned_project_name: {finetuned_project_name}")
    print(f"  n_indices: {n_indices}")

    # Create folder
    finetuned_project_experiment_folder = os.path.join(
        experiment_folder, finetuned_project_name
    )
    os.makedirs(finetuned_project_experiment_folder, exist_ok=True)

    # Save meta data
    all_cfg = {
        "data_folder": data_folder,
        "experiment_folder": experiment_folder,
        "pretrained_project_name_weak": pretrained_project_name_weak,
        "pretrained_project_name_strong": pretrained_project_name_strong,
        "finetuned_project_name": finetuned_project_name,
        "n_indices": n_indices,
        "training_cfg_finetune": training_cfg_finetune,
    }
    training_cfg_path = os.path.join(
        finetuned_project_experiment_folder, "training_cfg.json"
    )
    with open(training_cfg_path, "w") as f:
        json.dump(all_cfg, f, indent=4)

    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    finetuning_pairs = [
        (weak_size, model_sizes[j])
        for i, weak_size in enumerate(model_sizes)
        for j in range(i + 1, len(model_sizes))  # strong_size > weak_size index
    ]

    all_jobs: list[Job] = []
    for index in range(n_indices):
        print(f"\n--- Preparing jobs for Index {index} ---")
        jobs_for_index = 0
        for weak_size, strong_size in finetuning_pairs:
            job_args = (
                data_folder,
                experiment_folder,
                pretrained_project_name_weak,
                pretrained_project_name_strong,
                finetuned_project_name,
                weak_size,
                strong_size,
                copy.deepcopy(training_cfg_finetune),
                index,
            )
            all_jobs.append(Job(function=run_tictactoe_finetuning_task, args=job_args))
            jobs_for_index += 1
        print(
            f"  Prepared {jobs_for_index} finetuning jobs for Index {index}. Total jobs scheduled so far: {len(all_jobs)}"
        )

    parallel_gpu_executor = ParallelGpuExecutor()
    parallel_gpu_executor.submit_jobs(all_jobs)
    print("\nPretraining script finished.")


if __name__ == "__main__":
    data_folder = "/homes/55/bwilop/wsg/data/tictactoe/"
    experiment_folder = "/homes/55/bwilop/wsg/experiments/tictactoe/"

    pretrained_project_name_weak = "tictactoe_pretraining"
    # pretrained_project_name_strong = "tictactoe_pretraining"
    pretrained_project_name_strong = "tictactoe_pretraining_random"

    # finetuned_project_name = "tictactoe_finetuning4"
    # finetuned_project_name = "tictactoe_finetuning_use_best_val_step4"
    finetuned_project_name = "tictactoe_finetuning_random4"
    # finetuned_project_name = "tictactoe_finetuning_use_best_val_step_random4"

    training_cfg_finetune = get_training_cfg_finetune()
    training_cfg_finetune["use_best_val_checkpoint"] = False

    n_indices = 10
    main_finetune_multi_index(
        data_folder,
        experiment_folder,
        pretrained_project_name_weak,
        pretrained_project_name_strong,
        finetuned_project_name,
        n_indices,
        training_cfg_finetune,
    )
