import torch as t
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
from datetime import datetime
import wandb
from copy import deepcopy

from wsg_games.tictactoe.train.train import rearrange, log_generating_game_wandb, evaluate_model
from wsg_games.tictactoe.data import random_sample_tictactoe_data, TicTacToeData
from wsg_games.tictactoe.train.save_load_models import save_model, load_model, load_finetuned_model_get_matching_files, load_finetuned_model
from wsg_games.tictactoe.game import Goal


def quick_evaluation(name, model, test_data):
    model.eval()
    with t.no_grad():
        test_sample = random_sample_tictactoe_data(test_data, 20000)
        test_logits = model(test_sample.games_data)
        weak_loss   = cross_entropy(rearrange(test_logits), rearrange(test_sample.weak_goals_labels)).item()
        strong_loss = cross_entropy(rearrange(test_logits), rearrange(test_sample.strong_goals_labels)).item()
        print(name)
        print("weak_loss: ", weak_loss)
        print("strong_loss: ", strong_loss)


def evaluate_model_finetuning(model, train_games, train_labels, test_games, test_labels, loss_fn, n_samples=1000):
    train_indices = t.randperm(train_games.size(0))[:n_samples]
    test_indices  = t.randperm(test_games.size(0))[:n_samples]
    train_sample = train_games[train_indices]
    train_sample_labels = train_labels[train_indices]
    test_sample = test_games[test_indices]
    test_sample_labels = test_labels[test_indices]

    model.eval()
    with t.no_grad():
        train_logits = model(train_sample)
        test_logits  = model(test_sample)
        train_loss = loss_fn(rearrange(train_logits), rearrange(train_sample_labels)).item()
        test_loss  = loss_fn(rearrange(test_logits), rearrange(test_sample_labels)).item()

    wandb.log({
        "finetune/train": train_loss,
        "finetune/test": test_loss,
    })


def finetune_strong_with_weak(project_name: str, 
                              weak_model, weak_model_size: str, strong_model, strong_model_size: str, 
                              weak_train_data: TicTacToeData, val_data: TicTacToeData, test_data: TicTacToeData, training_cfg: dict):
    """
    Early stopping by checkpointing after every optimizer step, then early stop with patience 1.
    """
    lr = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    max_epochs = training_cfg.get("max_epochs")
    batch_size = training_cfg.get("batch_size")
    early_stopping_patience = 1

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
    optimizer =  t.optim.AdamW(strong_model.parameters(), lr=lr, weight_decay=weight_decay)

    wandb.finish()  # in case a previous run is still active
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"experiment_{weak_model_size}_{strong_model_size}_{timestamp}"
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
            "early_stopping_patience": early_stopping_patience,
            "batch_size": batch_size,
        }
    )

    # Finetuning loop: train strong_model to match the weak_model predictions
    log_generating_game_wandb(strong_model)
    evaluate_model(strong_model, weak_train_data, test_data, loss_fn, n_samples=20000)
    evaluate_model_finetuning(strong_model, weak_train_data.games_data, train_weak_labels, test_data.games_data, test_weak_labels, loss_fn)

    best_val_loss_epoch = float('inf')
    best_val_loss_batch = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    n_datapoints_since_last_evaluation = 0
    n_datapoints_since_last_generation_evaluation = 0
    for epoch in tqdm(range(max_epochs), desc="Training epochs", position=0, dynamic_ncols=True):
        # -------------------------
        # Training Phase (mini-batch loop)
        # -------------------------
        strong_model.train()
        for games, labels in tqdm(train_loader, desc="Training batches", leave=False, position=1, dynamic_ncols=True):
            optimizer.zero_grad()
            logits = strong_model(games)

            loss = loss_fn(rearrange(logits), rearrange(labels))
            loss.backward()
            optimizer.step()

            n_datapoints_since_last_evaluation += batch_size
            if n_datapoints_since_last_evaluation > 0:
                n_datapoints_since_last_evaluation = 0
                evaluate_model(strong_model, weak_train_data, test_data, loss_fn, n_samples=20000)
                evaluate_model_finetuning(strong_model, weak_train_data.games_data, train_weak_labels, test_data.games_data, test_weak_labels, loss_fn)

            n_datapoints_since_last_generation_evaluation += batch_size
            if n_datapoints_since_last_generation_evaluation > 10000:
                n_datapoints_since_last_generation_evaluation = 0
                log_generating_game_wandb(strong_model)

            # Get best model after each batch
            strong_model.eval()
            with t.no_grad():
                val_logits = strong_model(val_data.games_data)
                val_loss = loss_fn(rearrange(val_logits), rearrange(val_data.weak_goals_labels)).item()
            
            wandb.log({"val/val_loss_batch": val_loss})
            if val_loss < best_val_loss_batch:
                best_val_loss_batch = val_loss
                best_model_state = strong_model.state_dict()

        # Early stopping after epochs
        if val_loss < best_val_loss_epoch:
            best_val_loss_epoch = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        wandb.log({"val/val_loss_epoch": val_loss, "val/best_epoch": best_epoch})

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_val_loss_epoch:.4f}")
            break

    if best_model_state is not None:
        strong_model.load_state_dict(best_model_state)

    run_id = wandb.run.id
    wandb.finish()
    return strong_model, experiment_name, run_id


def finetune_sweep(pretrained_project_name: str, finetuned_project_name: str, experiment_folder: str,
                   weak_finetune_data: TicTacToeData, val_data: TicTacToeData, test_data: TicTacToeData,
                   training_cfg: dict):
    model_sizes = ["nano", "micro", "mini", "small", "medium", "large", "huge"]
    
    for i, weak_size in enumerate(model_sizes):
        weak_model = load_model(pretrained_project_name, weak_size, Goal.WEAK_GOAL, experiment_folder)
        if not weak_model:
            print(f"Weak model of size {weak_size} not found, skipping.")
            continue
        
        for j in range(i + 1, len(model_sizes)):
            strong_size = model_sizes[j]
            matching_files = load_finetuned_model_get_matching_files(finetuned_project_name, weak_size, strong_size, experiment_folder)
            if matching_files:
                print(f"Finetuned {strong_size} to {weak_size} already exists. Skipping ...")
            else:
                strong_model = load_model(pretrained_project_name, strong_size, Goal.STRONG_GOAL, experiment_folder)
                if not strong_model:
                    print(f"Strong model of size {strong_size} not found, skipping.")
                    continue

                finetuned_model = deepcopy(strong_model)
                print(f"Finetuning: weak model ({weak_size}) -> strong model ({strong_size})")
                
                finetuned_model, experiment_name, run_id = finetune_strong_with_weak(
                    finetuned_project_name,
                    weak_model, weak_size,
                    finetuned_model, strong_size,
                    weak_finetune_data, val_data, test_data,
                    training_cfg
                )
                # Save the finetuned model.
                save_model(finetuned_model, run_id, finetuned_project_name, experiment_name, experiment_folder)


