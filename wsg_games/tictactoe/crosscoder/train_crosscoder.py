import torch as t
from typing import Iterator, Any
import transformer_lens.utils as utils
import json
from torch.utils.data import DataLoader, TensorDataset
import datetime
from typing import Iterator, Any
from dictionary_learning.dictionary_learning.cache import *
from datetime import datetime
import wandb

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.crosscoder import (
    BatchTopKCrossCoderTrainer,
)


def get_training_cfg_cross_coder():
    training_cfg_cross_coder = {
        "learning_rate": 1e-3,
        "max_steps": 20000,
        "validate_every_n_steps": 1000,
        "batch_size": 64,
        "expansion_factor": 4,  # 32 in https://arxiv.org/pdf/2504.02922?
        "k": 10,
    }
    return training_cfg_cross_coder


def multi_epoch_dataloader_iterator(
    dataloader: DataLoader, total_steps_to_yield: int
) -> Iterator[Any]:
    """
    A generator that yields batches from a DataLoader repeatedly until
    total_steps_to_yield is reached. Re-shuffles if dataloader.shuffle=True.
    """
    # Edge cases
    if total_steps_to_yield == 0:  # No steps
        return
    try:
        if len(dataloader) == 0 and total_steps_to_yield > 0:  # Empty dataloader
            print(
                "Warning: DataLoader is empty, but total_steps_to_yield > 0. No steps will run."
            )
            return
    except TypeError:  # no __len__
        pass

    steps_yielded = 0
    while steps_yielded < total_steps_to_yield:
        num_batches_this_epoch = 0
        for batch in dataloader:  # DataLoader shuffles here if its shuffle=True
            if steps_yielded >= total_steps_to_yield:
                return
            yield batch
            steps_yielded += 1
            num_batches_this_epoch += 1

        # Safeguard, if the dataloader gets empty for any reason, it would be an infinite loop otherwise
        if num_batches_this_epoch == 0 and steps_yielded < total_steps_to_yield:
            print("Warning: DataLoader became empty before all steps were yielded.")
            return


def train_crosscoder(
    model_0_name: str,
    model_1_name: str,
    index: int,
    train_activations_stor_dir_model_0: str,
    val_activations_stor_dir_model_0: str,
    train_activations_stor_dir_model_1: str,
    val_activations_stor_dir_model_1: str,
    layer: int,
    training_cfg_cross_coder: dict,
    wandb_entity: str,
    device: t.device,
    crosscoder_folder: str,
) -> None:
    # Save arguments
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"experiment_{index}_{model_0_name}_{model_1_name}_{timestamp}"
    save_dir = crosscoder_folder + "checkpoints/" + experiment_name
    train_crosscoder_args = {
        "model_0_name": model_0_name,
        "model_1_name": model_1_name,
        "index": index,
        "layer": layer,
        "training_cfg_cross_coder": training_cfg_cross_coder,
        "data_path": {
            "train_activations_stor_dir_model_0": train_activations_stor_dir_model_0,
            "val_activations_stor_dir_model_0": val_activations_stor_dir_model_0,
            "train_activations_stor_dir_model_1": train_activations_stor_dir_model_1,
            "val_activations_stor_dir_model_1": val_activations_stor_dir_model_1,
        },
    }
    save_dir = os.path.join(crosscoder_folder, "checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, "train_crosscoder_args.json"), "w", encoding="utf-8"
    ) as json_file:
        json.dump(train_crosscoder_args, json_file)

    # Data (not loaded in memory yet)
    train_dataset = PairedActivationCache(
        train_activations_stor_dir_model_0,
        train_activations_stor_dir_model_1,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg_cross_coder["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    print(f"Training on {len(train_dataset)} token activations.")
    val_dataset = PairedActivationCache(
        val_activations_stor_dir_model_0,
        val_activations_stor_dir_model_1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    print(f"Validating on {len(val_dataset)} token activations.")

    # Training config
    activation_dim = train_dataset[0].shape[1]
    dictionary_size = training_cfg_cross_coder["expansion_factor"] * activation_dim
    print(f"Activation dim: {activation_dim}")
    print(f"Dictionary size: {dictionary_size}")
    k = training_cfg_cross_coder["k"]
    assert 0 < k <= dictionary_size
    lr = training_cfg_cross_coder["learning_rate"]
    max_steps = training_cfg_cross_coder["max_steps"]

    # Top level of trainer_cfg: BatchTopKCrossCoderTrainer
    # dict_class_kwargs: CrossCoder
    trainer_cfg = {
        "trainer": BatchTopKCrossCoderTrainer,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "device": str(device),
        "wandb_name": experiment_name + f"L{layer}-k{k:.1e}-lr{lr:.0e}",  #
        "steps": max_steps,  # Also used below, I think because it is needed here for learning rate and in trainSAE just for loop termination
        "k": k,  # 'k' as a top-level argument for the trainer
        "layer": layer,  # Only for logging
        "lm_name": experiment_name,  # Only for logging
        # "dict_class_kwargs": {
        # },
    }
    # train the sparse autoencoder (SAE)
    wandb.finish()
    multi_epoch_train_dataloader = multi_epoch_dataloader_iterator(
        train_dataloader, max_steps
    )
    trainSAE(
        data=multi_epoch_train_dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=training_cfg_cross_coder["validate_every_n_steps"],
        validation_data=val_dataloader,
        use_wandb=True,
        wandb_entity=wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        save_dir=save_dir,
        steps=max_steps,
        save_steps=None,
    )
