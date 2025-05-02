import matplotlib.pyplot as plt
from copy import deepcopy
import torch as t
from torch.nn.functional import cross_entropy
from einops import rearrange

from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.data import random_sample_tictactoe_data
from wsg_games.tictactoe.train.train import run_full_training, rearrange
from wsg_games.tictactoe.train.save_load_models import load_model, load_model_get_matching_files, save_model
from wsg_games.tictactoe.train.create_models import count_parameters


def pretrain_models(experiment_folder:str, project_name: str, tictactoe_train_data, tictactoe_val_data, tictactoe_test_data, training_cfg, get_model_config, device: t.device) -> None:
    for model_size in ["nano", "micro", "mini", "small", "medium", "large", "huge"]:
        for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
            matching_files = load_model_get_matching_files(project_name, model_size, goal, experiment_folder, device)
            if not matching_files:
                model_cfg = get_model_config(model_size)
                model, experiment_name, run_id = run_full_training(project_name, model_size, goal, tictactoe_train_data, tictactoe_val_data, tictactoe_test_data, training_cfg, model_cfg)
                save_model(model, run_id, project_name, experiment_name, experiment_folder)
                model.cpu()
                del model
                t.cuda.empty_cache()

def compute_avg_loss(model, games_data, labels, batch_size=1000):
    total_loss = 0.0
    total_samples = 0

    for i in range(0, len(games_data), batch_size):
        batch_data = games_data[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        logits = model(batch_data)

        loss = cross_entropy(rearrange(logits), rearrange(batch_labels), reduction='sum')
        total_loss += loss.item()
        total_samples += len(rearrange(logits))

    return total_loss / total_samples

def plot_loss_pretrain_models(experiment_folder, project_name, test_data, device: t.device) -> None:
    minimal_loss_weak = 0.6561687588691711
    minimal_loss_strong = 0.5871079564094543

    data_by_goal = {
        Goal.WEAK_GOAL: {
            "params": [],
            "random_loss": [],
            "weak_loss": [],
            "strong_loss": []
        },
        Goal.STRONG_GOAL: {
            "params": [],
            "random_loss": [],
            "weak_loss": [],
            "strong_loss": []
        }
    }

    for model_size in ["nano", "micro", "mini", "small", "medium", "large", "huge", "gigantic"]:
        for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
            model = load_model(project_name, model_size, goal, experiment_folder, device)
            if not model:
                continue

            # Evaluate
            n_parameters = count_parameters(model)
            test_sample = random_sample_tictactoe_data(test_data, 10000)
            
            avg_random_loss = compute_avg_loss(model, test_sample.games_data, test_sample.random_move_labels)
            avg_weak_loss   = compute_avg_loss(model, test_sample.games_data, test_sample.weak_goals_labels)
            avg_strong_loss = compute_avg_loss(model, test_sample.games_data, test_sample.strong_goals_labels)

            data_by_goal[goal]["params"].append(n_parameters)
            data_by_goal[goal]["random_loss"].append(avg_random_loss)
            data_by_goal[goal]["weak_loss"].append(avg_weak_loss)
            data_by_goal[goal]["strong_loss"].append(avg_strong_loss)

            # Clean memory
            model.cpu()
            del model
            t.cuda.empty_cache()

    print(data_by_goal)

    # Create a figure with two subplots (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for models with WEAK_GOAL
    ax = axes[0]
    ax.plot(data_by_goal[Goal.WEAK_GOAL]["params"], data_by_goal[Goal.WEAK_GOAL]["random_loss"], 'o-', label="Random CE Loss")
    ax.plot(data_by_goal[Goal.WEAK_GOAL]["params"], data_by_goal[Goal.WEAK_GOAL]["weak_loss"], 's-', label="Weak CE Loss")
    ax.plot(data_by_goal[Goal.WEAK_GOAL]["params"], data_by_goal[Goal.WEAK_GOAL]["strong_loss"], 'd-', label="Strong CE Loss")
    ax.axhline(y=minimal_loss_weak, color='gray', linestyle='--', label="Min Achievable Weak Loss")

    ax.set_xscale('log')
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Params (Model trained on Goal: WEAK_GOAL)")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # Plot for models with STRONG_GOAL
    ax = axes[1]
    ax.plot(data_by_goal[Goal.STRONG_GOAL]["params"], data_by_goal[Goal.STRONG_GOAL]["random_loss"], 'o-', label="Random CE Loss")
    ax.plot(data_by_goal[Goal.STRONG_GOAL]["params"], data_by_goal[Goal.STRONG_GOAL]["weak_loss"], 's-', label="Weak CE Loss")
    ax.plot(data_by_goal[Goal.STRONG_GOAL]["params"], data_by_goal[Goal.STRONG_GOAL]["strong_loss"], 'd-', label="Strong CE Loss")
    ax.axhline(y=minimal_loss_strong, color='gray', linestyle='--', label="Min Achievable Strong Loss")

    ax.set_xscale('log')
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Params (Model trained on Goal: STRONG_GOAL)")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()
