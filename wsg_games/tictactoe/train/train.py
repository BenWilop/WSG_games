import torch as t
import einops
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
from torch.nn.functional import cross_entropy, softmax
import wandb
from datetime import datetime
from jaxtyping import Float
from transformer_lens import HookedTransformerConfig, HookedTransformer

from wsg_games.tictactoe.evals import evaluate_predictions, sample_games, eval_model
from wsg_games.tictactoe.data import TicTacToeData, random_sample_tictactoe_data
from wsg_games.tictactoe.game import Goal

def rearrange(tensor: t.Tensor) -> t.Tensor:
    """
    Flattens the first two dimensions (n_games and game_length) into one.
    This converts a tensor of shape [n_games, game_length, n_tokens]
    into [n_games * game_length, n_tokens] which is what F.cross_entropy expects.
    """
    return einops.rearrange(tensor, "batch seq token -> (batch seq) token")


def log_epoch_wandb(logits: Float[t.Tensor, "n_games game_length n_tokens"], data: TicTacToeData, loss_fn, folder) -> None:
  res = {}
  flat_logits = rearrange(logits)
  flat_weak_labels = rearrange(data.weak_goals_labels)
  flat_strong_labels = rearrange(data.strong_goals_labels)
  flat_random_labels = rearrange(data.random_move_labels)

  res = {}
  res[folder + 'weak_loss'] = loss_fn(flat_logits, flat_weak_labels).item()
  res[folder + 'strong_loss'] = loss_fn(flat_logits, flat_strong_labels).item()
  res[folder + 'random_loss'] = loss_fn(flat_logits, flat_random_labels).item()

  predictions = softmax(logits, -1)
  evaluation = evaluate_predictions(predictions, data)
  for metric, value in evaluation.items():
    res[folder + metric] = value
  wandb.log(res)


def log_generating_game_wandb(model, n_samples=20):
    samples = sample_games(model, 1, n_samples)
    evaluation = eval_model(samples)
    res = {}
    for metric, value in evaluation.items():
      res["generative/" + metric] = value
    wandb.log(res)


def evaluate_model(model, train_data, test_data, loss_fn, n_samples=100):
    model.eval()
    with t.no_grad():
        train_sample = random_sample_tictactoe_data(train_data, n_samples)
        train_logits = model(train_sample.games_data)
        log_epoch_wandb(train_logits, train_sample, loss_fn, "train/")

        test_sample = random_sample_tictactoe_data(test_data, n_samples)
        test_logits = model(test_sample.games_data)
        log_epoch_wandb(test_logits, test_sample, loss_fn, "test/")


def train_model(project_name: str, experiment_name: str, timestamp: str,
                model, goal: Goal, optimizer, loss_fn,
                train_data: TicTacToeData, test_data: TicTacToeData,
                epochs: int, batch_size: int) -> None:
    """Log train + test ~ every 1000 datapoints and generation every 50000"""
    log_generating_game_wandb(model)
    evaluate_model(model, train_data, test_data, loss_fn)

    # Dataloader for minibatches and shuffling
    train_dataset = TensorDataset(
        train_data.games_data,
        train_data.random_move_labels,
        train_data.weak_goals_labels,
        train_data.strong_goals_labels
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_datapoints_since_last_evaluation = 0
    n_datapoints_since_last_generation_evaluation = 0
    for epoch in tqdm(range(epochs), desc="Training epochs", position=0, dynamic_ncols=True):
        # -------------------------
        # Training Phase (mini-batch loop)
        # -------------------------
        model.train()
        for games, random_labels, weak_labels, strong_labels in tqdm(train_loader, desc="Training batches", leave=False, position=1, dynamic_ncols=True):
            match goal:
              case Goal.WEAK_GOAL:
                  labels = weak_labels
              case Goal.STRONG_GOAL:
                  labels = strong_labels
              case _:
                  raise ValueError(f"Unexpected goal {goal}")

            optimizer.zero_grad()
            logits = model(games)
            loss = loss_fn(rearrange(logits),
                           rearrange(labels))
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


def run_full_training(project_name, model_size: str, goal: Goal, train_data, test_data, training_cfg: dict, model_cfg: dict) -> None:
    lr = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    epochs = training_cfg.get("epochs")
    batch_size = training_cfg.get("batch_size")

    model = HookedTransformer(model_cfg).to(model_cfg.device)
    loss_fn = cross_entropy
    optimizer =  t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    wandb.finish()  # In case previous run did not get finished
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"experiment_{model_size}_{str(goal)}_{timestamp}"
    wandb.init(
        project=project_name,
        name=experiment_name,
        config = {
            "transformer_config": model_cfg.to_dict(),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "test_train_split": len(train_data.games_data) / (len(train_data.games_data) + len(test_data.games_data)),
            "epochs": epochs,
            "batch_size": batch_size,
        })
    run_id = wandb.run.id
    train_model(project_name, experiment_name, timestamp,
        model, goal, optimizer, loss_fn,
        train_data, test_data, training_cfg.get("epochs"), batch_size=batch_size)

    wandb.finish()

    return model, experiment_name, run_id