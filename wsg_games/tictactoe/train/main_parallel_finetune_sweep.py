#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_finetune_sweep.py
import time
import queue
import copy
import torch as t
import torch.multiprocessing as mp
from multiprocessing import Manager

from wsg_games.tictactoe.train.finetune import finetune_strong_with_weak
from wsg_games.tictactoe.train.save_load_models import (
    load_model,
    save_model,
    load_finetuned_model_get_matching_files,
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

def worker(gpu_id: int,
           pretrained_project_name, finetuned_project_name, experiment_folder,
           weak_data, val_data, test_data, training_cfg,
           task_queue):
    # bind this process to its GPU
    t.cuda.set_device(gpu_id)
    print(f"[Worker {gpu_id}] on {t.cuda.get_device_name(gpu_id)}")
    device = t.device(f"cuda:{gpu_id}")

    weak_data = move_tictactoe_data_to_device(weak_data, device=device)
    val_data = move_tictactoe_data_to_device(val_data, device=device)
    test_data = move_tictactoe_data_to_device(test_data, device=device)
    while True:
        try:
            weak_size, strong_size = task_queue.get_nowait()
        except queue.Empty:
            print(f"[Worker {gpu_id}] no more tasks, exiting.")
            break

        print(f"[Worker {gpu_id}] starting task {weak_size}→{strong_size}")
        # skip if already done
        if load_finetuned_model_get_matching_files(finetuned_project_name, weak_size, 
                                                   strong_size, experiment_folder):
            print(f"[Worker {gpu_id}] already finetuned {weak_size}→{strong_size}, skipping")
            continue

        # load pretrained weak & strong
        weak_model = load_model(pretrained_project_name, weak_size, Goal.WEAK_GOAL, experiment_folder, device)
        strong_model = load_model(pretrained_project_name, strong_size, Goal.STRONG_GOAL, experiment_folder, device)
        if weak_model is None or strong_model is None:
            print(f"[Worker {gpu_id}] missing pretrained model for {weak_size} or {strong_size}, skipping")
            continue
        weak_model = move_to_and_update_config(weak_model, device)
        strong_model = move_to_and_update_config(strong_model, device)
        weak_model = weak_model.to(device)
        strong_model = strong_model.to(device)

        print(f"{gpu_id} weak_data", weak_data.games_data.device)
        print(f"{gpu_id} val_data", val_data.games_data.device)
        print(f"{gpu_id} test_data", test_data.games_data.device)
        for name, param in weak_model.named_parameters():
            print(f"{gpu_id} weak_model {name:30s} -> {param.device}")
        for name, buf in weak_model.named_buffers():
            print(f"{gpu_id} weak_model buffer {name:30s} -> {buf.device}")
        for name, param in strong_model.named_parameters():
            print(f"{gpu_id} strong_model {name:30s} -> {param.device}")
        for name, buf in strong_model.named_buffers():
            print(f"{gpu_id} strong_model buffer {name:30s} -> {buf.device}")

        # perform finetune
        finetuned_model, exp_name, run_id = finetune_strong_with_weak(
            finetuned_project_name,
            weak_model, weak_size,
            strong_model, strong_size,
            weak_data, val_data, test_data,
            training_cfg
        )

        # save
        save_model(finetuned_model, run_id, finetuned_project_name, exp_name, experiment_folder)
        print(f"[Worker {gpu_id}] finished {weak_size}→{strong_size} (run {run_id})")

        # free GPU cache
        t.cuda.empty_cache()


def main():
    pretrained_project_name = "tictactoe_pretrained_reverse_rule_no_overlap_split_start_third_200k"
    finetuned_project_name = "finetune_sweep_test_parallel"
    data_folder = '/homes/55/bwilop/wsg/data/'
    experiment_folder = '/homes/55/bwilop/wsg/experiments/'

    # Data has to be on CPU, otherwise worker starts working on GPU:0 by default.
    tictactoe_data = cache_tictactoe_data_random(data_folder + 'tictactoe_data_random_STRONG_RULE_REVERSE_RULE.pkl', device=None)  # None so data lives on CPU.
    _, tictactoe_weak_finetune_data, tictactoe_val_data, tictactoe_test_data = train_test_split_tictactoe_first_two_moves_no_overlap(tictactoe_data, 42, 15, 5, 10, 1234)
    tictactoe_weak_finetune_data = create_hard_label_tictactoe_data(tictactoe_weak_finetune_data, num_samples=1)
    tictactoe_val_data = create_hard_label_tictactoe_data(tictactoe_val_data, num_samples=1)
    tictactoe_test_data = create_hard_label_tictactoe_data(tictactoe_test_data, num_samples=1)

    training_cfg = get_training_cfg()

    # 2) build task list
    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    tasks = [
        (weak_size, model_sizes[j])
        for i, weak_size in enumerate(model_sizes)
        for j in range(i+1, len(model_sizes))
    ]
    print(f"Total tasks: {len(tasks)}")

    # 3) launch one worker per GPU
    ngpus = t.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    # shared queue
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
                pretrained_project_name, finetuned_project_name, experiment_folder,
                tictactoe_weak_finetune_data, tictactoe_val_data, tictactoe_test_data, training_cfg,
                task_queue
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All finetune tasks completed.")

if __name__ == "__main__":
    main()
