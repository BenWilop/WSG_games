from transformer_lens import HookedTransformerConfig, HookedTransformer
import torch.nn as nn


def get_training_cfg():
    training_cfg = {
        "learning_rate": 1e-3,  # 1e-4,
        "weight_decay": 1e-4,  # 1e-5,
        "max_epochs": 1000,
        "early_stopping_patience": 3,
        "batch_size": 64,
    }
    return training_cfg


def get_training_cfg_finetune():
    training_cfg_finetune = {
        "learning_rate": 1e-5,  # 1e-3
        "weight_decay": 1e-2,  # 1e-4
        "max_epochs": 1000,
        "early_stopping_patience_after_each_optimizer_step": 10000,  # 100
        "use_best_val_checkpoint": True,
        "batch_size": 64,
    }
    return training_cfg_finetune


def get_model_sizes():
    model_sizes = {}
    model_sizes["nano"] = {
        "n_layers": 1,
        "n_heads": 1,
        "d_model": 1,
        "d_head": 1,
        "d_mlp": 4,
    }
    model_sizes["micro"] = {
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 4,
        "d_head": 2,
        "d_mlp": 16,
    }
    model_sizes["mini"] = {
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 8,
        "d_head": 2,
        "d_mlp": 32,
    }

    model_sizes["small"] = {
        "n_layers": 3,
        "n_heads": 4,
        "d_model": 16,
        "d_head": 4,
        "d_mlp": 64,
    }
    model_sizes["medium"] = {
        "n_layers": 4,
        "n_heads": 8,
        "d_model": 32,
        "d_head": 4,
        "d_mlp": 128,
    }
    model_sizes["large"] = {
        "n_layers": 5,
        "n_heads": 8,
        "d_model": 64,
        "d_head": 8,
        "d_mlp": 256,
    }

    model_sizes["huge"] = {
        "n_layers": 6,
        "n_heads": 16,
        "d_model": 128,
        "d_head": 8,
        "d_mlp": 512,
    }
    # model_sizes["gigantic"] = {"n_layers": 7, "n_heads": 16, "d_model": 256, "d_head": 16, "d_mlp": 1024}
    return model_sizes


def get_model_config(size: str):
    common_params = {
        "act_fn": "relu",
        "normalization_type": "LN",
        "d_vocab": 11,
        "d_vocab_out": 10,
        "n_ctx": 10,
        "init_weights": True,
        "seed": 1337,
    }
    model_sizes = get_model_sizes()
    specific = model_sizes[size]
    return HookedTransformerConfig(**specific, **common_params)


def format_integer_scientific(n: float) -> str:
    s = f"{n:.1e}"
    return s.replace("e+", " * 10^")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_model_ratios(model_sizes: dict[str, int], get_model_config) -> None:
    last = 0
    ratios = []
    for model_size in model_sizes.keys():
        cfg = get_model_config(model_size)
        mod = HookedTransformer(cfg).to(cfg.device)
        n = count_parameters(mod)
        if last != 0:
            ratios.append(n / last)
        last = n
        print(model_size, format_integer_scientific(n))
        del mod
    print("Ratio of consecutive model-sizes: ", ratios)
