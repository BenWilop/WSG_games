import torch as t

from wsg_games.tictactoe.train.train import run_full_training
from wsg_games.tictactoe.train.save_load_models import (
    save_model,
    load_model_get_matching_files,
)
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_training_cfg,
    get_model_config,
)
from wsg_games.tictactoe.data import (
    cache_tictactoe_data_random,
    train_test_split_tictactoe_first_two_moves_no_overlap,
    create_hard_label_tictactoe_data,
    move_tictactoe_data_to_device,
)


def run_tictactoe_training_task(
    device: t.device,
    goal: Goal,
    model_size: str,
    project_name: str,
    experiment_folder: str,
    tictactoe_train_data_cpu,
    tictactoe_val_data_cpu,
    tictactoe_test_data_cpu,
    training_cfg,
):
    """Function to be executed by worker process."""
    # skip if already done
    if load_model_get_matching_files(project_name, model_size, goal, experiment_folder):
        print(
            f"[Task on GPU {str(device)}] already pretrained {goal} and {model_size}, skipping"
        )
        return

    # Move data to device
    print(
        f"[Task on GPU {str(device)}] preparing for {goal} and {model_size} on device {device}"
    )
    tictactoe_train_data = move_tictactoe_data_to_device(
        tictactoe_train_data_cpu, device=device
    )
    tictactoe_val_data = move_tictactoe_data_to_device(
        tictactoe_val_data_cpu, device=device
    )
    tictactoe_test_data = move_tictactoe_data_to_device(
        tictactoe_test_data_cpu, device=device
    )

    # Train
    print(f"[Task on GPU {str(device)}] starting training for {goal} and {model_size}")
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
    save_model(model, run_id, project_name, experiment_name, experiment_folder)
    print(
        f"[Task on GPU {str(device)}] finished {goal} and {model_size} (run {run_id})"
    )
