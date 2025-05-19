import matplotlib.pyplot as plt
import torch as t
from torch.nn.functional import cross_entropy
from einops import rearrange
import pandas as pd
import seaborn as sns
import os

from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.data import (
    random_sample_tictactoe_data,
    load_split_data,
    move_tictactoe_data_to_device,
)
from wsg_games.tictactoe.train.train import run_full_training, rearrange
from wsg_games.tictactoe.train.save_load_models import (
    load_model,
    load_model_get_matching_files,
    save_model,
)
from wsg_games.tictactoe.train.create_models import count_parameters


def pretrain_models(
    experiment_folder: str,
    project_name: str,
    tictactoe_train_data,
    tictactoe_val_data,
    tictactoe_test_data,
    training_cfg,
    get_model_config,
    device: t.device,
) -> None:
    for model_size in ["nano", "micro", "mini", "small", "medium", "large", "huge"]:
        for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
            matching_files = load_model_get_matching_files(
                project_name, model_size, goal, experiment_folder
            )
            if not matching_files:
                model_cfg = get_model_config(model_size)
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
                save_model(
                    model, run_id, project_name, experiment_name, experiment_folder
                )
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

        loss = cross_entropy(
            rearrange(logits), rearrange(batch_labels), reduction="sum"
        )
        total_loss += loss.item()
        total_samples += len(rearrange(logits))

    return total_loss / total_samples


def plot_loss_pretrain_models(
    data_folder: str,
    experiment_folder: str,
    project_name: str,
    device: t.device,
    indices: int | list[int | None] | None,
    save_path: str | None = None,
) -> None:
    minimal_loss_weak = 0.34
    minimal_loss_strong = 0.32

    # Indices
    if type(indices) == int:
        indices = [indices]
    elif indices is None:
        max_idx = -1
        for i in range(100):
            if not load_model_get_matching_files(
                project_name, "nano", Goal.WEAK_GOAL, experiment_folder, i
            ):
                max_idx = i
                break
        indices = [i for i in range(max_idx)]

    print("Indices: ", indices)

    # Evaluate models
    results = []
    for index in indices:
        _, _, _, tictactoe_test_data = load_split_data(
            data_folder, device=device, index=index
        )
        tictactoe_test_data = move_tictactoe_data_to_device(
            tictactoe_test_data, device=device
        )
        for model_size in [
            "nano",
            "micro",
            "mini",
            "small",
            "medium",
            "large",
            "huge",
        ]:
            for goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
                model = load_model(
                    project_name, model_size, goal, experiment_folder, device, index
                )
                if not model:
                    print("Missing: ", index, model_size, goal)
                    continue

                # Evaluate
                n_parameters = count_parameters(model)

                avg_random_loss = compute_avg_loss(
                    model,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.random_move_labels,
                )
                avg_weak_loss = compute_avg_loss(
                    model,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.weak_goals_labels,
                )
                avg_strong_loss = compute_avg_loss(
                    model,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.strong_goals_labels,
                )

                results.append(
                    {
                        "index": index
                        if index is not None
                        else -1,  # -1 as placeholder for None to make df easier
                        "goal": goal,
                        "model_size": model_size,
                        "n_parameters": n_parameters,
                        "avg_random_loss": avg_random_loss,
                        "avg_weak_loss": avg_weak_loss,
                        "avg_strong_loss": avg_strong_loss,
                    }
                )

                # Clean memory
                model.cpu()
                del model
                t.cuda.empty_cache()

    if len(results) == 0:
        print("No models found.")
        return

    # Plot
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left (Performance on weak rule)
    ax = axes[0]
    df_weak = df[df["goal"] == Goal.WEAK_GOAL]
    sns.lineplot(  # Random
        data=df_weak,
        x="n_parameters",
        y="avg_random_loss",
        label="Random CE Loss",
        marker="o",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    sns.lineplot(  # Weak
        data=df_weak,
        x="n_parameters",
        y="avg_weak_loss",
        label="Weak CE Loss",
        marker="s",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    sns.lineplot(  # Strong
        data=df_weak,
        x="n_parameters",
        y="avg_strong_loss",
        label="Strong CE Loss",
        marker="d",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    ax.axhline(
        y=minimal_loss_weak,
        color="gray",
        linestyle="--",
        label="Min Achievable Weak Loss",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 4)
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Loss (μ ± σ over indices)")
    ax.set_title(f"Loss vs. Params (Model trained on Goal: {Goal.WEAK_GOAL.value})")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    # Right (Performance on strong rule)
    ax = axes[1]
    df_strong = df[df["goal"] == Goal.STRONG_GOAL]
    sns.lineplot(  # Random
        data=df_strong,
        x="n_parameters",
        y="avg_random_loss",
        label="Random CE Loss",
        marker="o",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    sns.lineplot(  # Weak
        data=df_strong,
        x="n_parameters",
        y="avg_weak_loss",
        label="Weak CE Loss",
        marker="s",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    sns.lineplot(  # Strong
        data=df_strong,
        x="n_parameters",
        y="avg_strong_loss",
        label="Strong CE Loss",
        marker="d",
        linestyle="-",
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    ax.axhline(
        y=minimal_loss_strong,
        color="gray",
        linestyle="--",
        label="Min Achievable Strong Loss",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 4)
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Loss (μ ± σ over indices)")
    ax.set_title(f"Loss vs. Params (Model trained on Goal: {Goal.STRONG_GOAL.value})")
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "pretrain.png")
        try:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()
