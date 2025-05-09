#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_parallel_finetune_sweep.py
import time
import queue
import copy
import torch as t
import torch.multiprocessing as mp
from multiprocessing import Manager

from wsg_games.tictactoe.train.train import run_full_training
from wsg_games.tictactoe.train.save_load_models import (
    load_model,
    save_model,
    load_model_get_matching_files,
)
from transformer_lens.utilities.devices import move_to_and_update_config
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_training_cfg,
)
from wsg_games.tictactoe.data import (
    cache_tictactoe_data_random,
    train_test_split_tictactoe_first_two_moves_no_overlap,
    create_hard_label_tictactoe_data,
    move_tictactoe_data_to_device
)
from wsg_games.tictactoe.train.create_models import get_model_config


def worker(gpu_id: int,
           project_name, experiment_folder,
           tictactoe_train_data, tictactoe_val_data, tictactoe_test_data, training_cfg,
           task_queue):
    # Set GPU
    t.cuda.set_device(gpu_id)
    print(f"[Worker {gpu_id}] on {t.cuda.get_device_name(gpu_id)}")
    device = str(t.device(f"cuda:{gpu_id}"))

    tictactoe_train_data = move_tictactoe_data_to_device(tictactoe_train_data, device=device)
    tictactoe_val_data = move_tictactoe_data_to_device(tictactoe_val_data, device=device)
    tictactoe_test_data = move_tictactoe_data_to_device(tictactoe_test_data, device=device)

    # Run until queue empty
    while True:
        try:
            goal, model_size = task_queue.get_nowait()
        except queue.Empty:
            print(f"[Worker {gpu_id}] no more tasks, exiting.")
            break

        print(f"[Worker {gpu_id}] starting task {goal} and {model_size}")
        # skip if already done
        if load_model_get_matching_files(project_name, model_size, goal, experiment_folder):
            print(f"[Worker {gpu_id}] already pretrained {goal} and {model_size}, skipping")
            continue

        model_cfg = get_model_config(model_size)
        model, experiment_name, run_id = run_full_training(project_name, 
                                                           model_size, 
                                                           goal, 
                                                           tictactoe_train_data, 
                                                           tictactoe_val_data, 
                                                           tictactoe_test_data, 
                                                           training_cfg, 
                                                           model_cfg)

        # save
        save_model(model, run_id, project_name, experiment_name, experiment_folder)
        print(f"[Worker {gpu_id}] finished {goal} and {model_size} (run {run_id})")
        t.cuda.empty_cache()


def main_pretrain():
    project_name = "tictactoe_no_diagonal_rule_test"
    data_folder = '/homes/55/bwilop/wsg/data/'
    experiment_folder = '/homes/55/bwilop/wsg/experiments/'

    # Data 
    # has to be on CPU, otherwise worker starts working on GPU:0 by default.
    tictactoe_data = cache_tictactoe_data_random(data_folder + 'tictactoe_data_random_STRONG_RULE_REVERSE_RULE.pkl', device="cpu")  # None so data lives on CPU.
    tictactoe_train_data, _, tictactoe_val_data, tictactoe_test_data = train_test_split_tictactoe_first_two_moves_no_overlap(tictactoe_data, 42, 15, 5, 10, 1234)
    tictactoe_train_data = create_hard_label_tictactoe_data(tictactoe_train_data, num_samples=1)
    tictactoe_val_data = create_hard_label_tictactoe_data(tictactoe_val_data, num_samples=1)
    tictactoe_test_data = create_hard_label_tictactoe_data(tictactoe_test_data, num_samples=1)

    training_cfg = get_training_cfg()

    # Tasts
    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    goals = [Goal.WEAK_GOAL, Goal.STRONG_GOAL]
    tasks = [
        (goal, model_size) 
        for goal in goals
        for model_size in model_sizes
    ]
    print(f"Total tasks: {len(tasks)}")

    # Start multiprocessing
    ngpus = t.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    task_queue = manager.Queue()
    for task in tasks:
        task_queue.put(task)

    processes = []
    for gpu in range(ngpus):
        p = mp.Process(
            target=worker,
            args=(
                gpu,
                project_name, experiment_folder,
                tictactoe_train_data, tictactoe_val_data, tictactoe_test_data, training_cfg,
                task_queue
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All finetune tasks completed.")

if __name__ == "__main__":
    main_pretrain()
