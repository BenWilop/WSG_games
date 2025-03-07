from transformer_lens import HookedTransformerConfig, HookedTransformer
import torch.nn as nn

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