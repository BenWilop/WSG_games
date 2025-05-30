import torch as t
import einops
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
from torch.nn.functional import cross_entropy, softmax
import wandb
from datetime import datetime
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.utilities.devices import move_to_and_update_config

from wsg_games.tictactoe.evals import evaluate_predictions, sample_games, eval_model
from wsg_games.tictactoe.data import TicTacToeData, random_sample_tictactoe_data
from wsg_games.tictactoe.game import Goal
from wsg_games.meta import Game, game_to_ignore_first_n_moves


def rearrange(tensor: t.Tensor, game: Game = Game.TICTACTOE) -> t.Tensor:
    """
    Prepares data from game dimensionality to 2D for evaluation.

    Flattens the first two dimensions (n_games and game_length) into one.
    This converts a tensor of shape [n_games, game_length, n_tokens]
    into [n_games * game_length, n_tokens] which is what a loss_fn such as F.cross_entropy expects.
    """
    ignore_first_n_moves = game_to_ignore_first_n_moves(game)
    return einops.rearrange(
        tensor[:, ignore_first_n_moves:, :], "batch seq token -> (batch seq) token"
    )


def log_epoch_wandb(
    logits: Float[t.Tensor, "n_games game_length n_tokens"],
    data: TicTacToeData,
    loss_fn,
    folder,
) -> None:
    """Logs loss and accuracy."""
    res: dict[str, float] = {}
    flat_logits = rearrange(logits)
    flat_weak_labels = rearrange(data.weak_goals_labels)
    flat_strong_labels = rearrange(data.strong_goals_labels)
    flat_random_labels = rearrange(data.random_move_labels)

    res = {}
    res[folder + "weak_loss"] = loss_fn(flat_logits, flat_weak_labels).item()
    res[folder + "strong_loss"] = loss_fn(flat_logits, flat_strong_labels).item()
    res[folder + "random_loss"] = loss_fn(flat_logits, flat_random_labels).item()

    predictions = softmax(logits, -1)
    evaluation = evaluate_predictions(predictions, data)
    for metric, value in evaluation.items():
        res[folder + metric] = value
    wandb.log(res)


def log_generating_game_wandb(model, n_samples=20):
    """Generates games using the model and logs the performance"""
    samples = sample_games(model, 1, n_samples)
    evaluation = eval_model(samples)
    res = {}
    for metric, value in evaluation.items():
        res["generative/" + metric] = value
    wandb.log(res)


def evaluate_model(model, train_data, test_data, loss_fn, n_samples=1000):
    """Logs train and test set loss and performance."""
    model.eval()
    with t.no_grad():
        train_sample = random_sample_tictactoe_data(train_data, n_samples)
        train_logits = model(train_sample.games_data)
        log_epoch_wandb(train_logits, train_sample, loss_fn, "train/")

        test_sample = random_sample_tictactoe_data(test_data, n_samples)
        test_logits = model(test_sample.games_data)
        log_epoch_wandb(test_logits, test_sample, loss_fn, "test/")


def train_model(
    model,
    goal: Goal,
    optimizer,
    loss_fn,
    train_data: TicTacToeData,
    val_data: TicTacToeData,
    test_data: TicTacToeData,
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
) -> None:
    """Log train + test ~ every 1000 datapoints and generation every 50000"""
    log_generating_game_wandb(model)
    evaluate_model(model, train_data, test_data, loss_fn)

    # Dataloaders for minibatches and shuffling
    train_dataset = TensorDataset(
        train_data.games_data,
        train_data.random_move_labels,
        train_data.weak_goals_labels,
        train_data.strong_goals_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    n_datapoints_since_last_evaluation = 0
    n_datapoints_since_last_generation_evaluation = 0
    for epoch in tqdm(
        range(max_epochs), desc="Training epochs", position=0, dynamic_ncols=True
    ):
        # -------------------------
        # Training Phase (mini-batch loop)
        # -------------------------
        model.train()
        for games, _, weak_labels, strong_labels in tqdm(
            train_loader,
            desc="Training batches",
            leave=False,
            position=1,
            dynamic_ncols=True,
        ):
            match goal:
                case Goal.WEAK_GOAL:
                    labels = weak_labels
                case Goal.STRONG_GOAL:
                    labels = strong_labels
                case _:
                    raise ValueError(f"Unexpected goal {goal}")

            optimizer.zero_grad()
            logits = model(games)
            loss = loss_fn(rearrange(logits), rearrange(labels))
            loss.backward()
            optimizer.step()

            n_datapoints_since_last_evaluation += batch_size
            if n_datapoints_since_last_evaluation > 1000:
                n_datapoints_since_last_evaluation = 0
                evaluate_model(model, train_data, test_data, loss_fn)

            n_datapoints_since_last_generation_evaluation += batch_size
            if n_datapoints_since_last_generation_evaluation > 50000:
                n_datapoints_since_last_generation_evaluation = 0
                log_generating_game_wandb(model)

        # Early stopping
        model.eval()
        with t.no_grad():
            match goal:
                case Goal.WEAK_GOAL:
                    labels = val_data.weak_goals_labels
                case Goal.STRONG_GOAL:
                    labels = val_data.strong_goals_labels
                case _:
                    raise ValueError(f"Unexpected goal {goal}")
            logits = model(val_data.games_data)
            val_loss = loss_fn(rearrange(logits), rearrange(labels)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()  # checkpoint best model
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            wandb.log({"val/val_loss": val_loss, "val/best_epoch": best_epoch})

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


def run_full_training(
    project_name,
    model_size: str,
    goal: Goal,
    train_data,
    val_data,
    test_data,
    training_cfg: dict,
    model_cfg,
) -> tuple[HookedTransformer, str, str]:
    lr = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    max_epochs = training_cfg.get("max_epochs")
    early_stopping_patience = training_cfg.get("early_stopping_patience")
    batch_size = training_cfg.get("batch_size")

    model = HookedTransformer(model_cfg)
    model = move_to_and_update_config(model, model_cfg.device)

    loss_fn = cross_entropy
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    wandb.finish()  # In case previous run did not get finished
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"{model_size}_{str(goal)}_{timestamp}"
    n_train_data = len(train_data.games_data)
    n_val_data = len(val_data.games_data)
    n_test_data = len(test_data.games_data)
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "transformer_config": model_cfg.to_dict(),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "n_train_data": n_train_data,
            "n_val_data": n_val_data,
            "n_test_data": n_test_data,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
        },
    )
    run_id = wandb.run.id
    train_model(
        model,
        goal,
        optimizer,
        loss_fn,
        train_data,
        val_data,
        test_data,
        max_epochs,
        early_stopping_patience,
        batch_size=batch_size,
    )

    wandb.finish()

    return model, experiment_name, run_id
