import matplotlib.pyplot as plt
from copy import deepcopy
import torch as t
from torch.nn.functional import cross_entropy
from einops import rearrange

from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.data import random_sample_tictactoe_data
from wsg_games.tictactoe.create_models import get_model_config
from wsg_games.tictactoe.train.train import run_full_training
from wsg_games.tictactoe.train.save_load_models import load_model, load_model_get_matching_files, save_model
from wsg_games.tictactoe.analyse_data import count_parameters, 


def pretrain_models(experiment_folder:str, project_name: str, tictactoe_train_data, tictactoe_test_data, training_cfg, model_size_to_epochs) -> None:
    for model_size in ["nano", "micro", "mini", "small", "medium", "large"]:
        for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
            matching_files = load_model_get_matching_files(project_name, model_size, goal, experiment_folder)
            if not matching_files:
                adapted_training_cfg = deepcopy(training_cfg)
                adapted_training_cfg["epochs"] = model_size_to_epochs[model_size]

                model_cfg = get_model_config(model_size)
                model, experiment_name, run_id = run_full_training(project_name, model_size, goal, tictactoe_train_data, tictactoe_test_data, adapted_training_cfg, model_cfg)
                save_model(model, run_id, project_name, experiment_name, experiment_folder)
                model.cpu()
                del model
                t.cuda.empty_cache()


def plot_loss_pretrain_models(experiment_folder, project_name, test_data):
    minimal_loss_weak = 0.4463610351085663
    minimal_loss_strong = 0.23728151619434357

    # Initialize dictionaries to store the data for each goal type.
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

    # Iterate over model sizes and both goal types.
    for model_size in ["nano", "micro", "mini", "small", "medium", "large"]: 
        for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
            model = load_model(project_name, model_size, goal, experiment_folder)
            if not model:
                continue

            # Evaluate
            n_parameters = count_parameters(model)
            test_sample = random_sample_tictactoe_data(test_data, 1000)
            test_logits = model(test_sample.games_data)
            random_loss = cross_entropy(rearrange(test_logits), rearrange(test_sample.random_move_labels)).item()
            weak_loss   = cross_entropy(rearrange(test_logits), rearrange(test_sample.weak_goals_labels)).item()
            strong_loss = cross_entropy(rearrange(test_logits), rearrange(test_sample.strong_goals_labels)).item()

            # Save the computed values in the appropriate goal category
            data_by_goal[goal]["params"].append(n_parameters)
            data_by_goal[goal]["random_loss"].append(random_loss)
            data_by_goal[goal]["weak_loss"].append(weak_loss)
            data_by_goal[goal]["strong_loss"].append(strong_loss)

            model.cpu()
            del model
            t.cuda.empty_cache()

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
    ax.axhline(y=minimal_loss_strong, color='gray', linestyle='--', label="Min Achievable StrongLoss")

    ax.set_xscale('log')
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Params (Model trained on Goal: STRONG_GOAL)")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()