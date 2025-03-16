import torch as t
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
import wandb

from wsg_games.tictactoe.train.train import rearrange, log_generating_game_wandb, evaluate_model
from wsg_games.tictactoe.data import random_sample_tictactoe_data

def quick_evaluation(name, model, test_data):
    model.eval()
    with t.no_grad():
        test_sample = random_sample_tictactoe_data(test_data, 1000)
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


def finetune_strong_with_weak(project_name, experiment_name, weak_model, strong_model, train_data, test_data, training_cfg: dict):
    lr = training_cfg.get("learning_rate")
    weight_decay = training_cfg.get("weight_decay")
    epochs = training_cfg.get("epochs")
    batch_size = training_cfg.get("batch_size")

    # Compute weak labels using weak_model predictions
    weak_model.eval()
    with t.no_grad():
        train_logits = weak_model(train_data.games_data)
        train_weak_labels = softmax(train_logits, dim=-1)
        # train_weak_labels = F.one_hot(train_logits.argmax(dim=-1), num_classes=train_logits.shape[-1]).float()
        # train_weak_labels = F.one_hot(train_data.weak_goals_labels.argmax(dim=-1), num_classes=train_logits.shape[-1]).float()
        test_logits = weak_model(test_data.games_data)
        test_weak_labels = softmax(test_logits, dim=-1) 
        # test_weak_labels = F.one_hot(test_logits.argmax(dim=-1), num_classes=test_logits.shape[-1]).float()

    train_dataset = TensorDataset(train_data.games_data, train_weak_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = cross_entropy
    optimizer =  t.optim.AdamW(strong_model.parameters(), lr=lr, weight_decay=weight_decay)

    wandb.finish()  # in case a previous run is still active
    wandb.init(
        project=project_name,
        name=experiment_name,
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size
        }
    )
    run_id = wandb.run.id

    alpha = None  # 0.7
    temperature = 1
    print("alpha: ", alpha)
    print("temperature: ", temperature)

    # Finetuning loop: train strong_model to match the weak_model predictions
    log_generating_game_wandb(strong_model)
    evaluate_model(strong_model, train_data, test_data, loss_fn, n_samples=25000)
    evaluate_model_finetuning(strong_model, train_data.games_data, train_weak_labels, test_data.games_data, test_weak_labels, loss_fn)
    n_datapoints_since_last_evaluation = 0
    n_datapoints_since_last_generation_evaluation = 0
    for epoch in tqdm(range(epochs), desc="Training epochs", position=0, dynamic_ncols=True):
        # -------------------------
        # Training Phase (mini-batch loop)
        # -------------------------
        strong_model.train()
        for games, labels in tqdm(train_loader, desc="Training batches", leave=False, position=1, dynamic_ncols=True):
            optimizer.zero_grad()
            logits = strong_model(games)

            # confidence loss
            if alpha:
                strong_model_predictions = softmax(logits / temperature, dim=-1)
                labels = alpha*labels + (1-alpha)*strong_model_predictions

            loss = loss_fn(rearrange(logits), rearrange(labels))
            loss.backward()
            optimizer.step()

            n_datapoints_since_last_evaluation += batch_size
            if n_datapoints_since_last_evaluation > 0:
                n_datapoints_since_last_evaluation = 0
                evaluate_model(strong_model, train_data, test_data, loss_fn, n_samples=25000)
                evaluate_model_finetuning(strong_model, train_data.games_data, train_weak_labels, test_data.games_data, test_weak_labels, loss_fn)

            n_datapoints_since_last_generation_evaluation += batch_size
            if n_datapoints_since_last_generation_evaluation > 10000:
                n_datapoints_since_last_generation_evaluation = 0
                log_generating_game_wandb(strong_model)




