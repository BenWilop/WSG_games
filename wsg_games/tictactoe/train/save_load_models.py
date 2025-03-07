import os
import sys
import glob
import torch as t
from wsg_games.tictactoe.game import Goal


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
    model = t.load(latest_file)
    return model