import torch as t
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
from datetime import datetime
import wandb
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import os
import numpy as np


from wsg_games.tictactoe.train.train import (
    rearrange,
    log_generating_game_wandb,
    evaluate_model,
)
from wsg_games.tictactoe.data import (
    load_split_data,
    move_tictactoe_data_to_device,
    TicTacToeData,
)
from wsg_games.tictactoe.train.save_load_models import (
    save_model,
    load_model,
    load_finetuned_model_get_matching_files,
    load_finetuned_model,
)
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.pretrain import compute_avg_loss
from wsg_games.tictactoe.train.create_models import count_parameters
from wsg_games.tictactoe.data import TicTacToeData, random_sample_tictactoe_data


def quick_evaluation(name, model, test_data):
    """Prints weak and strong test loss."""
    model.eval()
    with t.no_grad():
        test_sample = random_sample_tictactoe_data(test_data, 20000)
        test_logits = model(test_sample.games_data)
        weak_loss = cross_entropy(
            rearrange(test_logits), rearrange(test_sample.weak_goals_labels)
        ).item()
        strong_loss = cross_entropy(
            rearrange(test_logits), rearrange(test_sample.strong_goals_labels)
        ).item()
        print(name)
        print("weak_loss: ", weak_loss)
        print("strong_loss: ", strong_loss)
    return weak_loss, strong_loss


def evaluate_model_finetuning(
    model, train_games, train_labels, test_games, test_labels, loss_fn, n_samples=1000
):
    """Logs test and train set loss."""
    train_indices = t.randperm(train_games.size(0))[:n_samples]
    test_indices = t.randperm(test_games.size(0))[:n_samples]
    train_sample = train_games[train_indices]
    train_sample_labels = train_labels[train_indices]
    test_sample = test_games[test_indices]
    test_sample_labels = test_labels[test_indices]

    model.eval()
    with t.no_grad():
        train_logits = model(train_sample)
        test_logits = model(test_sample)
        train_loss = loss_fn(
            rearrange(train_logits), rearrange(train_sample_labels)
        ).item()
        test_loss = loss_fn(
            rearrange(test_logits), rearrange(test_sample_labels)
        ).item()

    wandb.log(
        {
            "finetune/train": train_loss,
            "finetune/test": test_loss,
        }
    )


def finetune_strong_with_weak(
    project_name: str,
    weak_model,
    weak_model_size: str,
    strong_model,
    strong_model_size: str,
    weak_train_data: TicTacToeData,
    val_data: TicTacToeData,
    test_data: TicTacToeData,
    training_cfg_finetune: dict,
) -> tuple:
    """
    Early stopping by checkpointing after every optimizer step, then early stop with patience 1.
    """
    lr = training_cfg_finetune.get("learning_rate")
    weight_decay = training_cfg_finetune.get("weight_decay")
    early_stopping_patience_after_each_optimizer_step = training_cfg_finetune.get(
        "early_stopping_patience_after_each_optimizer_step"
    )
    use_best_val_checkpoint = training_cfg_finetune.get("use_best_val_checkpoint")
    max_epochs = training_cfg_finetune.get("max_epochs")
    batch_size = training_cfg_finetune.get("batch_size")

    # Compute weak labels using weak_model predictions
    weak_model.eval()
    with t.no_grad():
        train_logits = weak_model(weak_train_data.games_data)
        train_weak_labels = softmax(train_logits, dim=-1)
        # train_weak_labels = F.one_hot(train_logits.argmax(dim=-1), num_classes=train_logits.shape[-1]).float()
        # train_weak_labels = F.one_hot(weak_train_data.weak_goals_labels.argmax(dim=-1), num_classes=train_logits.shape[-1]).float()
        test_logits = weak_model(test_data.games_data)
        test_weak_labels = softmax(test_logits, dim=-1)
        # test_weak_labels = F.one_hot(test_logits.argmax(dim=-1), num_classes=test_logits.shape[-1]).float()

    train_dataset = TensorDataset(weak_train_data.games_data, train_weak_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = cross_entropy
    optimizer = t.optim.AdamW(
        strong_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    wandb.finish()  # in case a previous run is still active
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"{weak_model_size}_{strong_model_size}_{timestamp}"
    n_weak_train_data = len(weak_train_data.games_data)
    n_val_data = len(val_data.games_data)
    n_test_data = len(test_data.games_data)
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "n_weak_train_data": n_weak_train_data,
            "n_val_data": n_val_data,
            "n_test_data": n_test_data,
            "max_epochs": max_epochs,
            "early_stopping_patience_after_each_optimizer_step": early_stopping_patience_after_each_optimizer_step,
            "use_best_val_checkpoint": use_best_val_checkpoint,
            "batch_size": batch_size,
        },
    )

    # Finetuning loop: train strong_model to match the weak_model predictions
    log_generating_game_wandb(strong_model)
    evaluate_model(strong_model, weak_train_data, test_data, loss_fn, n_samples=20000)
    evaluate_model_finetuning(
        strong_model,
        weak_train_data.games_data,
        train_weak_labels,
        test_data.games_data,
        test_weak_labels,
        loss_fn,
    )

    early_stop_triggered = False
    best_val_loss_optimizer_step = float("inf")
    optimizer_step = 0
    best_optimizer_step = 0
    patience_counter = 0
    best_model_state = None
    n_datapoints_since_last_evaluation = 0
    n_datapoints_since_last_generation_evaluation = 0
    for epoch in tqdm(
        range(max_epochs), desc="Training epochs", position=0, dynamic_ncols=True
    ):
        if early_stop_triggered:
            break
        # -------------------------
        # Training Phase (mini-batch loop)
        # -------------------------
        strong_model.train()
        for games, labels in tqdm(
            train_loader,
            desc="Training batches",
            leave=False,
            position=1,
            dynamic_ncols=True,
        ):
            optimizer.zero_grad()
            logits = strong_model(games)

            loss = loss_fn(rearrange(logits), rearrange(labels))
            loss.backward()
            optimizer.step()

            n_datapoints_since_last_evaluation += batch_size
            if n_datapoints_since_last_evaluation > 0:
                n_datapoints_since_last_evaluation = 0
                evaluate_model(
                    strong_model, weak_train_data, test_data, loss_fn, n_samples=20000
                )
                evaluate_model_finetuning(
                    strong_model,
                    weak_train_data.games_data,
                    train_weak_labels,
                    test_data.games_data,
                    test_weak_labels,
                    loss_fn,
                )

            n_datapoints_since_last_generation_evaluation += batch_size
            if n_datapoints_since_last_generation_evaluation > 10000:
                n_datapoints_since_last_generation_evaluation = 0
                log_generating_game_wandb(strong_model)

            # Get best model after each batch
            strong_model.eval()
            with t.no_grad():
                val_logits = strong_model(val_data.games_data)
                val_loss = loss_fn(
                    rearrange(val_logits), rearrange(val_data.weak_goals_labels)
                ).item()

            # Early stopping after optimizer step
            optimizer_step += 1
            if val_loss < best_val_loss_optimizer_step:
                best_val_loss_optimizer_step = val_loss
                best_optimizer_step = optimizer_step
                patience_counter = 0
                if use_best_val_checkpoint:
                    best_model_state = strong_model.state_dict()
            else:
                patience_counter += 1

            wandb.log(
                {
                    "val/val_loss_batch": val_loss,
                    "val/val_loss_optimizer_step": val_loss,
                    "val/best_optimizer_step": best_optimizer_step,
                }
            )

            if patience_counter >= early_stopping_patience_after_each_optimizer_step:
                print(
                    f"Early stopping triggered at step {optimizer_step}. Best epoch was {best_optimizer_step} with val loss {best_val_loss_optimizer_step:.4f}"
                )
                early_stop_triggered = True
                break

    if best_model_state is not None:
        strong_model.load_state_dict(best_model_state)

    run_id = wandb.run.id
    wandb.finish()
    return strong_model, experiment_name, run_id


def _plot_wsg_side_by_side(
    df: pd.DataFrame,
    model_sizes: list[str],
    weak_size_to_color: dict[str, tuple],
    save_path: str | None,
) -> None:
    """
    Creates a side-by-side plot for 'before' and 'after' finetuning,
    showing the mean WSG gap with standard error of the mean (SEM).

    The y-axis uses a symmetric log scale for consistent styling.
    """

    df["wsg_gap_before_finetuning"] /= 100
    df["wsg_gap_after_finetuning"] /= 100

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle("WSG Gap Analysis: Before vs. After Finetuning (Mean ± SEM)", y=1.02)

    # --- Helper function for advanced axis formatting ---
    def _format_axis(ax, title):
        ax.set_xscale("log")
        ax.set_xlabel("Strong Model Parameters (log scale)")
        ax.set_title(title)

        # --- Y-axis Symlog Scaling and Formatting from Reference ---
        ax.set_yscale("symlog", linthresh=1.0)

        # Determine y-axis range from all data points
        all_vals = pd.concat(
            [df["wsg_gap_before_finetuning"], df["wsg_gap_after_finetuning"]]
        ).dropna()
        y_min, y_max = all_vals.min(), all_vals.max()
        y_margin = (y_max - y_min) * 0.1
        # y_range = (min(-10, y_min - y_margin), max(120, y_max + y_margin))
        y_range = (-50, 1)
        ax.set_ylim(y_range)

        # Add horizontal reference lines
        ax.axhline(0, color="red", linestyle="-", linewidth=0.8)
        # ax.axhline(1, color="blue", linestyle=":", linewidth=1) # Original line for 100%

        # Add shaded regions for the log parts of the symlog scale
        ax.axhspan(1, y_range[1], alpha=0.1, color="gray", zorder=0)
        ax.axhspan(y_range[0], -1, alpha=0.1, color="gray", zorder=0)

        # --- Custom Y-axis Ticks for Symlog Scale ---
        linear_ticks = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        log_ticks_pos = [2, 5, 10, 20, 50, 100, 200, 500]
        log_ticks_neg = [-2, -5, -10, -20, -50, -100, -200, -500]

        all_ticks = sorted(
            list(
                set(
                    [t for t in log_ticks_neg if t >= y_range[0]]
                    + [-1]
                    + linear_ticks
                    + [1]
                    + [t for t in log_ticks_pos if t <= y_range[1]]
                )
            )
        )

        tick_labels = [
            f"{tick:.1f}" if -1 < tick < 1 and tick != 0 else f"{tick}"
            for tick in all_ticks
        ]
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(tick_labels)

        ax.grid(True, which="both", ls="--", alpha=0.3)

    # --- Main Plotting Loop ---
    for weak_size in model_sizes:
        df_weak = df[df["weak_size"] == weak_size]
        if df_weak.empty:
            continue

        # Aggregate data: calculate mean and standard error of the mean
        agg_df = (
            df_weak.groupby("n_parameters_strong")
            .agg(
                mean_before=("wsg_gap_before_finetuning", "mean"),
                sem_before=(
                    "wsg_gap_before_finetuning",
                    lambda x: np.std(x, ddof=1) / np.sqrt(np.size(x))
                    if np.size(x) > 1
                    else 0,
                ),
                mean_after=("wsg_gap_after_finetuning", "mean"),
                sem_after=(
                    "wsg_gap_after_finetuning",
                    lambda x: np.std(x, ddof=1) / np.sqrt(np.size(x))
                    if np.size(x) > 1
                    else 0,
                ),
            )
            .reset_index()
            .sort_values("n_parameters_strong")
        )

        if agg_df.empty:
            continue

        color = weak_size_to_color.get(weak_size)
        x = agg_df["n_parameters_strong"]

        # --- Plot "Before Finetuning" (Left) ---
        y_mean_before = agg_df["mean_before"]
        y_sem_before = agg_df["sem_before"]
        ax_before.plot(
            x,
            y_mean_before,
            marker="o",
            linestyle="-",
            color=color,
            label=f"Weak: {weak_size}",
        )
        ax_before.fill_between(
            x,
            y_mean_before - y_sem_before,
            y_mean_before + y_sem_before,
            color=color,
            alpha=0.2,
        )

        # --- Plot "After Finetuning" (Right) ---
        y_mean_after = agg_df["mean_after"]
        y_sem_after = agg_df["sem_after"]
        ax_after.plot(x, y_mean_after, marker="o", linestyle="-", color=color)
        ax_after.fill_between(
            x,
            y_mean_after - y_sem_after,
            y_mean_after + y_sem_after,
            color=color,
            alpha=0.2,
        )

    # --- Final Formatting ---
    _format_axis(ax_before, "Before Finetuning")
    _format_axis(ax_after, "After Finetuning")
    ax_before.set_ylabel("Recovered % (PGR)")

    handles, labels = ax_before.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Weak Model Size",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(model_sizes),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        # File suffix is now constant as the scale is no longer an option
        full_save_path = os.path.join(save_path, "side_by_side_plot.png")
        try:
            plt.savefig(full_save_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {full_save_path}")
        except Exception as e:
            print(f"Error saving plot to {full_save_path}: {e}")

    plt.show()


def plot_wsg_plot(
    df: pd.DataFrame,
    model_sizes: list[str],
    aggregate_data: bool,
    y_axis_log: bool,
    save_path: str | None,
) -> None:
    n_weak_sizes = len(model_sizes)
    num_cols_individual = 2
    num_rows_individual = (
        (
            n_weak_sizes
            + num_cols_individual
            - 1
            - 1  # num_cols_individual - 1 to round up and second -1 because last plot is empty
        )
        // num_cols_individual
    )
    total_gs_rows = 1 + num_rows_individual  # 1 for summary row at the top

    # Figure size
    fig_height = 5 + 3.5 * num_rows_individual
    fig = plt.figure(figsize=(17, fig_height))
    gs = gridspec.GridSpec(total_gs_rows, num_cols_individual, figure=fig)

    # Color
    palette_colors = sns.color_palette(n_colors=max(1, n_weak_sizes))
    weak_size_to_color = {
        size: color for size, color in zip(model_sizes, palette_colors)
    }
    legend_handles, legend_labels = None, None  # To store for figure-level legend

    # --- 1. Top Row: Summary Plots ---
    # Top Left: After Finetuning (Overall)
    # Top Right: Before Finetuning (Overall)
    ax_summary_before = fig.add_subplot(gs[0, 0])
    ax_summary_after = fig.add_subplot(gs[0, 1], sharey=ax_summary_before)

    summary_plot_configs = [
        {
            "ax": ax_summary_before,
            "y_col": "wsg_gap_before_finetuning",
            "title": "Overall - WSG Gap Before Finetuning",
        },
        {
            "ax": ax_summary_after,
            "y_col": "wsg_gap_after_finetuning",
            "title": "Overall - WSG Gap After Finetuning",
        },
    ]

    for i, cfg in enumerate(summary_plot_configs):
        ax = cfg["ax"]
        if aggregate_data:
            sns.lineplot(
                data=df / 100,
                x="n_parameters_strong",
                y=cfg["y_col"],
                hue="weak_size",
                hue_order=model_sizes,
                palette=weak_size_to_color,
                marker="o",
                errorbar="sd",
                ax=ax,
            )
        else:
            sns.lineplot(
                data=df / 100,
                x="n_parameters_strong",
                y=cfg["y_col"],
                hue="weak_size",
                hue_order=model_sizes,
                palette=weak_size_to_color,
                marker="o",
                units="index",
                estimator=None,
                # linewidth=0.7,
                # alpha=0.7,
                ax=ax,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Strong Model Parameters (log scale)")
        ax.set_ylabel(
            f"Recovered % ({'μ ± σ' if aggregate_data else 'Individual Indices'})"
        )
        ax.set_title(cfg["title"])
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        ax.axhline(100, color="blue", linestyle=":", linewidth=0.8)
        if y_axis_log:
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_ylim(0, 100)
        ax.grid(True, which="both", ls="--")

        if ax.get_legend() is not None:
            if i == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    # --- 2. Subsequent Rows: Individual "After Finetuning" Plots ---
    if num_rows_individual > 0:
        for idx_weak, current_weak_size in enumerate(model_sizes[:-1]):
            gs_row = 1 + idx_weak // num_cols_individual  # Start from GridSpec row 1
            gs_col = idx_weak % num_cols_individual

            ax_individual = fig.add_subplot(
                gs[gs_row, gs_col], sharex=ax_summary_before, sharey=ax_summary_before
            )
            df_filtered = df[df["weak_size"] == current_weak_size].copy()
            plot_color = weak_size_to_color[current_weak_size]

            ax_individual.set_title(f"{current_weak_size} - After Finetuning")
            if (
                df_filtered.empty
                or df_filtered["wsg_gap_after_finetuning"].isnull().all()
            ):
                ax_individual.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax_individual.transAxes,
                )
            else:
                if aggregate_data:
                    sns.lineplot(
                        data=df_filtered,
                        x="n_parameters_strong",
                        y="wsg_gap_after_finetuning",
                        marker="o",
                        errorbar="sd",
                        color=plot_color,
                        ax=ax_individual,
                        legend=False,
                    )
                else:
                    sns.lineplot(
                        data=df_filtered,
                        x="n_parameters_strong",
                        y="wsg_gap_after_finetuning",
                        marker="o",
                        units="index",
                        estimator=None,  # linewidth=0.7, alpha=0.7,
                        color=plot_color,
                        ax=ax_individual,
                        legend=False,
                    )

            ax_individual.set_xscale("log")
            if y_axis_log:
                ax_individual.set_yscale("symlog", linthresh=1.0)
            else:
                ax_individual.set_ylim(0, 100)
            ax_individual.axhline(0, color="black", linestyle=":", linewidth=0.8)
            ax_individual.axhline(100, color="blue", linestyle=":", linewidth=0.8)
            ax_individual.grid(True, which="both", ls="--")

            ax_individual.set_xlabel("Strong Model Parameters (log scale)")
            ax_individual.set_ylabel(
                "Recovered %"
            )  # Simpler Y label for individual plots

        # Hide any unused subplots in the individual plots grid
        for i in range(n_weak_sizes, num_rows_individual * num_cols_individual):
            gs_row_to_hide = 1 + i // num_cols_individual
            gs_col_to_hide = i % num_cols_individual
            if gs_row_to_hide < total_gs_rows and gs_col_to_hide < num_cols_individual:
                ax_to_hide = fig.add_subplot(gs[gs_row_to_hide, gs_col_to_hide])
                ax_to_hide.set_visible(False)

    # --- 3. Overall Figure Legend and Layout ---
    fig.suptitle(
        "WSG Gap Analysis: Overall Summary and Per-Weak-Model Detail (After Finetuning)",
        fontsize=18,
    )
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Weak Model Size (Overall Plots)",
            loc="upper center",  # Anchors the legend by its upper center point
            bbox_to_anchor=(0.5, 0.94),  # Positions the legend's anchor point:
            # 0.5 means horizontally centered in the figure.
            # 0.94 means 94% of the way up the figure (below the suptitle).
            # You might need to fine-tune this y-value (e.g., 0.93-0.95).
            ncol=max(
                1, (n_weak_sizes + 1) // 2
            ),  # Adjusted ncol for potentially better fit
        )

    bottom_margin = 0.1 if legend_handles else 0.03
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.96])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if y_axis_log:
            save_path = os.path.join(save_path, "finetune_log_scale.png")
        else:
            save_path = os.path.join(save_path, "finetune_linear_scale.png")
        try:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()


def plot_wsg_gap_finetuned_models(
    data_folder: str,
    experiment_folder: str,
    pretrained_project_name_weak: str,
    pretrained_project_name_strong: str,
    finetuned_project_name: str,
    device: t.device,
    indices: int | list[int | None] | None,
    aggregate_data: bool = True,
    save_path: str | None = None,
    overwrite: bool = False,
    plot_style: str = "grid",
) -> None:
    """
    Computes or loads cached WSG gap data and plots the results.

    Args:
        data_folder: Path to the data.
        experiment_folder: Path to the experiment logs and models.
        pretrained_project_name_weak: Project name for weak pretrained models.
        pretrained_project_name_strong: Project name for strong pretrained models.
        finetuned_project_name: Project name for finetuned models.
        device: The torch device to use for computation.
        indices: A single index or list of indices for data splits.
        aggregate_data: If True, plots mean and std dev; otherwise, individual lines.
        save_path: Directory to save plots and cache. Caching is disabled if None.
        overwrite: If True, forces re-computation and overwrites the existing cache.
    """
    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    df = None
    cache_file_path = None

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        cache_file_path = os.path.join(save_path, "wsg_results_cache.pkl")

    # --- 1. Load from Cache or Compute ---
    if cache_file_path and not overwrite and os.path.exists(cache_file_path):
        try:
            print(f"Loading cached results from {cache_file_path}...")
            df = pd.read_pickle(cache_file_path)
            # Ensure model_sizes list is consistent with the loaded data
            loaded_sizes = df["weak_size"].unique()
            model_sizes = [s for s in model_sizes if s in loaded_sizes]
            print("Successfully loaded results from cache.")
        except Exception as e:
            print(f"Could not load cache file: {e}. Recomputing...")
            df = None  # Ensure df is None to trigger recomputation

    if df is None:
        print("No cache found or overwrite=True, computing results...")
        # Indices
        if isinstance(indices, int):
            indices = [indices]
        elif indices is None:
            max_idx = -1
            for i in range(100):
                if not load_finetuned_model(
                    finetuned_project_name,
                    "nano",
                    "micro",
                    experiment_folder,
                    device,
                    i,
                ):
                    max_idx = i
                    break
            indices = list(range(max_idx))

        print("Indices: ", indices)

        # Evaluate models (pretrained)
        index_size_to_avg_weak_loss: dict[tuple[int, str], float] = {}
        index_size_to_n_parameters: dict[tuple[int, str], int] = {}
        for index in indices:
            _, _, _, tictactoe_test_data = load_split_data(
                data_folder, device=device, index=index
            )
            tictactoe_test_data = move_tictactoe_data_to_device(
                tictactoe_test_data, device=device
            )
            for model_size in model_sizes:
                model = load_model(
                    pretrained_project_name_weak,
                    model_size,
                    Goal.WEAK_GOAL,
                    experiment_folder,
                    device=device,
                    index=index,
                )
                if not model:
                    continue
                index_size_to_avg_weak_loss[(index, model_size)] = compute_avg_loss(
                    model,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.weak_goals_labels,
                )
                index_size_to_n_parameters[(index, model_size)] = count_parameters(
                    model
                )
                del model
                t.cuda.empty_cache()

        # Evaluate models (finetuned)
        finetuning_pairs = [
            (weak_size, model_sizes[j])
            for i, weak_size in enumerate(model_sizes)
            for j in range(i + 1, len(model_sizes))
        ]
        results = []
        for index in indices:
            _, _, _, tictactoe_test_data = load_split_data(
                data_folder, device=device, index=index
            )
            tictactoe_test_data = move_tictactoe_data_to_device(
                tictactoe_test_data, device=device
            )
            for weak_size, strong_size in finetuning_pairs:
                if (index, weak_size) not in index_size_to_avg_weak_loss or (
                    index,
                    strong_size,
                ) not in index_size_to_avg_weak_loss:
                    continue

                L_weak = index_size_to_avg_weak_loss[(index, weak_size)]
                L_strong_ceiling = index_size_to_avg_weak_loss[(index, strong_size)]

                # Strong model before finetuning
                strong_model_on_strong_rule = load_model(
                    pretrained_project_name_strong,
                    strong_size,
                    Goal.STRONG_GOAL,
                    experiment_folder,
                    device=device,
                    index=index,
                )
                if not strong_model_on_strong_rule:
                    continue
                L_weak_to_strong_before = compute_avg_loss(
                    strong_model_on_strong_rule,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.weak_goals_labels,
                )
                wsg_gap_before = (
                    (L_weak_to_strong_before - L_weak)
                    / (L_strong_ceiling - L_weak)
                    * 100
                )

                # Finetuned model
                finetuned_model = load_finetuned_model(
                    finetuned_project_name,
                    weak_size,
                    strong_size,
                    experiment_folder,
                    device=device,
                    index=index,
                )
                if not finetuned_model:
                    continue
                L_weak_to_strong_after = compute_avg_loss(
                    finetuned_model,
                    tictactoe_test_data.games_data,
                    tictactoe_test_data.weak_goals_labels,
                )
                wsg_gap_after = (
                    (L_weak_to_strong_after - L_weak)
                    / (L_strong_ceiling - L_weak)
                    * 100
                )

                results.append(
                    {
                        "index": index if index is not None else -1,
                        "weak_size": weak_size,
                        "strong_size": strong_size,
                        "n_parameters_weak": index_size_to_n_parameters[
                            (index, weak_size)
                        ],
                        "n_parameters_strong": index_size_to_n_parameters[
                            (index, strong_size)
                        ],
                        "wsg_gap_before_finetuning": wsg_gap_before,
                        "wsg_gap_after_finetuning": wsg_gap_after,
                    }
                )
                del finetuned_model, strong_model_on_strong_rule
                t.cuda.empty_cache()

        if not results:
            print("No models found or results generated.")
            return

        df = pd.DataFrame(results)
        # Save newly computed results to cache
        if cache_file_path:
            print(f"Saving results to cache at {cache_file_path}")
            df.to_pickle(cache_file_path)

    # --- 2. Plotting ---
    if df is not None and not df.empty:
        df.sort_values(by=["weak_size", "n_parameters_strong", "index"], inplace=True)

        # Create the color mapping dictionary here.
        # This is needed for the 'side_by_side' plot and is harmless for the 'grid' plot.
        palette_colors = sns.color_palette(n_colors=max(1, len(model_sizes)))
        weak_size_to_color = {
            size: color for size, color in zip(model_sizes, palette_colors)
        }

        # Use an if/elif block to call the correct function with the correct arguments
        if plot_style == "side_by_side":
            print("\nPlotting 'side_by_side' style (log scale y-axis)...")
            _plot_wsg_side_by_side(
                df=df,
                model_sizes=model_sizes,
                weak_size_to_color=weak_size_to_color,  # Pass the dictionary
                save_path=save_path,
            )

        elif plot_style == "grid":
            print("\nPlotting 'grid' style (log scale y-axis)...")
            plot_wsg_plot(
                df=df,
                model_sizes=model_sizes,
                aggregate_data=aggregate_data,  # Pass the boolean
                y_axis_log=True,
                save_path=save_path,
            )

            print("\nPlotting 'grid' style (linear scale y-axis)...")
            plot_wsg_plot(
                df=df,
                model_sizes=model_sizes,
                aggregate_data=aggregate_data,  # Pass the boolean
                y_axis_log=False,
                save_path=save_path,
            )
        else:
            print(
                f"Error: Unknown plot_style '{plot_style}'. Please use 'grid' or 'side_by_side'."
            )
            return
    else:
        print("No data available to plot.")
