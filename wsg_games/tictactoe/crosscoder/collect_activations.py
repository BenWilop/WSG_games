import torch as t
from jaxtyping import Float, Int
from torch import Tensor
import os

from dictionary_learning.cache import *
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

from wsg_games.meta import Game, game_to_ignore_first_n_moves
from wsg_games.tictactoe.train.train import rearrange
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.save_load_models import load_model, load_finetuned_model


def get_activations(
    model,
    tokenized_games: Float[Tensor, "n_games game_length"],
    layer_i: int,
) -> Float[Tensor, "n_games game_length d_model"]:
    activation_hook_name = utils.get_act_name("resid_post", layer_i)
    model.eval()
    _, cache = model.run_with_cache(tokenized_games)
    layer_activations = cache[activation_hook_name]
    return layer_activations


@t.no_grad()
def create_data_shards(
    game: Game,
    games_data: Float[Tensor, "n_games game_length"],
    model: HookedTransformer,
    store_dir: str,
    batch_size: int = 64,
    shard_size: int = 10**6,
    max_total_tokens: int = 10**8,
    overwrite: bool = False,
) -> None:
    ignore_first_n_moves = game_to_ignore_first_n_moves(game)
    io: str = "out"
    submodule_names = [f"layer_{layer_i}" for layer_i in range(model.cfg.n_layers)]

    token_cache = []
    activation_cache = [[] for _ in submodule_names]
    store_dirs = [
        os.path.join(store_dir, f"{submodule_names[layer_i]}_{io}")
        for layer_i in range(len(submodule_names))
    ]
    for dir in store_dirs:
        os.makedirs(dir, exist_ok=True)
    total_size = 0
    current_size = 0
    shard_count = 0

    # Check if shards already exist
    if os.path.exists(os.path.join(store_dirs[0], "shard_0.memmap")):
        print(f"Shards already exist in {store_dir}")
        if not overwrite:
            print("Instead, you can set overwrite=True to overwrite existing shards.")
            return
        else:
            print("Overwriting existing shards...")

    print("Collecting activations...")
    dataloader = DataLoader(games_data, batch_size=batch_size, shuffle=False)
    for games in tqdm(dataloader, desc="Collecting activations"):
        token_cache.append(games.cpu())  # [batch_size, game_length]

        for layer_i in range(len(submodule_names)):
            local_activations = rearrange(
                get_activations(model, games, layer_i),
                game=game,
            )  # (B x T) x D
            activation_cache[layer_i].append(local_activations.cpu())

        current_size += activation_cache[0][-1].shape[0]
        if current_size > shard_size:
            print(f"Storing shard {shard_count}...", flush=True)
            ActivationCache.collate_store_shards(
                store_dirs,
                shard_count,
                activation_cache,
                submodule_names,
                shuffle_shards=False,
                io=io,
                multiprocessing=False,
            )
            shard_count += 1
            total_size += current_size
            current_size = 0
            activation_cache = [[] for _ in submodule_names]

        if total_size > max_total_tokens:
            print("Max total tokens reached. Stopping collection.")
            break

    if current_size > 0:
        print(f"Storing shard {shard_count}...", flush=True)
        ActivationCache.collate_store_shards(
            store_dirs,
            shard_count,
            activation_cache,
            submodule_names,
            shuffle_shards=False,
            io=io,
            multiprocessing=False,
        )
        total_size += current_size
        shard_count += 1

    # Store tokens
    token_path = os.path.join(store_dir, "tokens.pt")
    print(f"Storing tokens in {token_path}")
    token_cache = t.cat(token_cache, dim=0)
    print("token_cache.shape: ", token_cache.shape)
    print("total_size: ", total_size)
    print(
        "token_cache.shape[1] - ignore_first_n_moves: ",
        token_cache.shape[1] - ignore_first_n_moves,
    )
    assert (
        token_cache.shape[0] * (token_cache.shape[1] - ignore_first_n_moves)
        == total_size
    )
    t.save(token_cache, token_path)  # [n_games, game_length]

    # store configs
    print(f"Storing configs.")
    for i, dir in enumerate(store_dirs):
        with open(os.path.join(dir, "config.json"), "w") as f:
            json.dump(
                {
                    "ignore_first_n_moves": ignore_first_n_moves,
                    "batch_size": batch_size,
                    "context_len": -1,
                    "shard_size": shard_size,
                    "d_model": model.cfg.d_model,
                    "shuffle_shards": False,
                    "io": io,
                    "total_size": total_size,
                    "shard_count": shard_count,
                    "store_tokens": True,
                },
                f,
            )
    ActivationCache.cleanup_multiprocessing()
    print(f"Finished collecting activations. Total size: {total_size}")


def get_activations_path(
    model_goal: Goal | None,
    weak_model_size: str | None,
    model_size: str,
    index: int,
    crosscoder_folder: str,
    train_val: str,
) -> str:
    assert model_goal is None or weak_model_size is None
    assert model_goal is not None or weak_model_size is not None
    if weak_model_size:
        postfix = "finetuned_through_" + weak_model_size
    elif model_goal in [Goal.WEAK_GOAL, Goal.STRONG_GOAL]:
        postfix = str(model_goal)
    else:
        raise ValueError(f"Invalid activations model goal: {model_goal}")
    return os.path.join(
        crosscoder_folder, "activations", f"{index}_{model_size}_{postfix}_" + train_val
    )


def compute_activations(
    game: Game,
    model_goal: Goal | None,
    project_name_pretrain: str | None,
    weak_model_size: str | None,
    project_name_finetune: str | None,
    model_size: str,
    index: int,
    crosscoder_folder: str,
    tictactoe_test_data: Float[Tensor, "n_games game_length"],
    tictactoe_val_data: Float[Tensor, "n_games game_length"],
    experiment_folder: str,
    device: t.device,
) -> None:
    """
    Trains crosscoder on the test data
    Validates on the validation data
    """
    # Either finetuned or pretrained
    bool_finetuned_model = (
        project_name_finetune is not None and weak_model_size is not None
    )
    bool_pretrained_model = project_name_pretrain is not None and model_goal is not None
    assert int(bool_finetuned_model) + int(bool_pretrained_model) == 1, (
        f"Finetuned XOR pretrained model must be provided."
    )

    # Models
    if bool_finetuned_model:
        model = load_finetuned_model(
            project_name_finetune,
            weak_model_size,
            model_size,
            experiment_folder,
            device,
            index,
        )
        print(project_name_finetune)
        print(weak_model_size)
        print(model_size)
        print(experiment_folder)
        print(device)
        print(index)
    else:
        model = load_model(
            project_name_pretrain,
            model_size,
            model_goal,
            experiment_folder,
            device=device,
            index=index,
        )

    # Run
    train_activations_path = None
    val_activations_path = None
    for train_val in ["train", "val"]:
        activations_path = get_activations_path(
            model_goal, weak_model_size, model_size, index, crosscoder_folder, train_val
        )
        if train_val == "train":
            train_activations_path = activations_path
            games_data = tictactoe_test_data
        elif train_val == "val":
            val_activations_path = activations_path
            games_data = tictactoe_val_data
        else:
            raise ValueError(f"Invalid train_val: {train_val}")
        create_data_shards(
            game,
            games_data,
            model,
            store_dir=activations_path,
            batch_size=64,
            shard_size=10**5,
            max_total_tokens=10**10,
            overwrite=False,
        )
    return train_activations_path, val_activations_path


def validate_activations(store_dirs: list[str]) -> None:
    """
    Assertions to verify that the activations of list_of_paths fit together,
    i.e. all were trained on same tokens and shapes are correct.
    """
    activation_cache_tuples = ActivationCacheTuple(*store_dirs)

    # The length of all ActivationCache must match in config and tensor
    n_tokens = len(activation_cache_tuples)
    for activation_cache in activation_cache_tuples.activation_caches:
        assert n_tokens == len(activation_cache)
        assert activation_cache.config["total_size"] == n_tokens
        activation_cache[n_tokens - 1]

    # The "ignore_first_n_moves" must be the same everywhere.
    ignore_first_n_moves = activation_cache_tuples.activation_caches[0].config[
        "ignore_first_n_moves"
    ]
    for activation_cache in activation_cache_tuples.activation_caches:
        assert activation_cache.config["ignore_first_n_moves"] == ignore_first_n_moves

    # All token values must be the same
    stacked_tokens = (
        activation_cache_tuples.tokens
    )  # [len(store_dir), n_games, game_length]
    token_0 = stacked_tokens[0]
    for i in range(stacked_tokens.shape[0]):
        assert th.equal(token_0, stacked_tokens[i]), (
            "Tokens of all activations must be the same to compare them with a Crosscoder."
        )


def get_list_of_games_from_paired_activation_cache(
    paired_activation_cache: PairedActivationCache, indices: Int[Tensor, "batch"]
) -> list[list[int]]:
    """
    Returns for each index (relating to the activations of paired_activation_cache)
    the subgame up until the move where the activation was created.
    So for the game [0,1,2,3,4,5,6,7,8,9], if we have as indices [3,6], because we
    relate to the activations collected after the 4th and 7th token, we return:
    list_of_games = [[0,1,2,3], [0,1,2,3,4,5,6]]
    """
    tokens = paired_activation_cache.tokens[0]  # [n_games, game_length]
    assert tokens.ndim == 2
    n_games, game_length = tokens.shape
    ignore_first_n_moves = paired_activation_cache.activation_cache_1.config[
        "ignore_first_n_moves"
    ]
    assert 0 <= ignore_first_n_moves < game_length
    n_activations = paired_activation_cache.activation_cache_1.config["total_size"]
    list_of_games: list[list[int]] = []
    for index in indices.cpu():
        assert 0 <= index < n_activations
        game_idx = index // (game_length - ignore_first_n_moves)
        idx_in_game = (
            index % (game_length - ignore_first_n_moves) + ignore_first_n_moves
        )
        assert 0 <= idx_in_game < game_length
        game = tokens[game_idx, 0:idx_in_game]
        list_of_games.append(game.tolist())

    return list_of_games
