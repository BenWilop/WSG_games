import os
import sys
import glob
import torch as t
from wsg_games.tictactoe.game import Goal
from transformer_lens import HookedTransformer


def save_model(model, run_id, project_name: str, experiment_name: str, experiment_folder: str) -> None:
    project_dir = f"{experiment_folder}/{project_name}"
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    os.makedirs(project_dir, exist_ok=True)

    file_name = f"{experiment_name}_{run_id}.pkl"
    file_path = os.path.join(project_dir, file_name)

    assert not os.path.exists(file_path)
    t.save(model, file_path)
    print(f"Model saved to {file_path}")


def load_model_get_matching_files(project_name: str, model_size: str, goal: Goal, experiment_folder: str):
    project_dir = f"{experiment_folder}/{project_name}"
    experiment_prefix = f"experiment_{model_size}_{str(goal)}_"
    pattern = os.path.join(project_dir, experiment_prefix + "*.pkl")
    matching_files = glob.glob(pattern)
    return matching_files


def load_model(project_name: str, model_size: str, goal: Goal, experiment_folder: str) -> t.nn.Module:
    matching_files = load_model_get_matching_files(project_name, model_size, goal, experiment_folder)

    if not matching_files:
        print(f"No model files found for size {model_size} and goal {goal}")
        return None

    # Pick the most recent file based on modification time.
    latest_file = max(matching_files, key=os.path.getmtime)
    print(f"Loading model from {latest_file}")
    with t.serialization.safe_globals({HookedTransformer}):
        model = t.load(latest_file, weights_only=False)
    return model


def load_finetuned_model_get_matching_files(project_name: str, weak_model_size: str, strong_model_size: str, experiment_folder: str):
    project_dir = os.path.join(experiment_folder, project_name)
    experiment_prefix = f"experiment_{weak_model_size}_{strong_model_size}_"
    pattern = os.path.join(project_dir, experiment_prefix + "*.pkl")
    matching_files = glob.glob(pattern)
    return matching_files

def load_finetuned_model(project_name: str, weak_model_size: str, strong_model_size: str, experiment_folder: str):
    matching_files = load_finetuned_model_get_matching_files(project_name, weak_model_size, strong_model_size, experiment_folder)
    if not matching_files:
        print(f"No finetuned model found for weak {weak_model_size} and strong {strong_model_size}")
        return None
    
    # Return newest model
    latest_file = max(matching_files, key=os.path.getmtime)
    with t.serialization.safe_globals({HookedTransformer}):
        finetuned_model = t.load(latest_file, weights_only=False)
    return finetuned_model