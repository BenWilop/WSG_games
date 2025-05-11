import torch as t
from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer


from wsg_games.tictactoe.data import TicTacToeData
from wsg_games.tictactoe.game import Board, State


def evaluate_predictions(
    predictions: Float[Tensor, "n_games game_length n_tokens"],
    tictactoe_data: TicTacToeData,
) -> dict[str, float]:
    """
    Returns dictionary metric -> value
    weak_accuracy, strong_accuracy
    - percentage weight of models predictions on correct moves
    - Example: label = [0.5, 0.5, 0], predictions = [0.1, 0.3, 0.6] -> accuracy = 0.4
    illegal_move_chance
    - Example: legal_moves = [1, 0, 1], predictions = [0.1, 0.3, 0.6] -> illegal_move_chance = 0.7
    """
    res = {}

    # Accuracy = percentage weight of models predictions on correct moves
    positive_weak_labels_mask = (
        tictactoe_data.weak_goals_labels > 0
    ).float()  # Shape: [n_games, game_length, n_tokens]
    predictions_on_positive_weak = predictions * positive_weak_labels_mask
    sum_of_weights_on_positive_weak = predictions_on_positive_weak.sum(
        dim=-1
    )  # Shape: [n_games, game_length]
    res["weak_accuracy"] = sum_of_weights_on_positive_weak.mean().item()

    positive_strong_labels_mask = (
        tictactoe_data.strong_goals_labels > 0
    ).float()  # Shape: [n_games, game_length, n_tokens]
    predictions_on_positive_strong = predictions * positive_strong_labels_mask
    sum_of_weights_on_positive_strong = predictions_on_positive_strong.sum(
        dim=-1
    )  # Shape: [n_games, game_length]
    res["strong_accuracy"] = sum_of_weights_on_positive_strong.mean().item()

    # Move is legal if the random move label is positive.
    illegal_mask = (
        tictactoe_data.random_move_labels == 0
    ).float()  # shape: [n_games, game_length, n_tokens]
    illegal_move_chance = (predictions * illegal_mask).sum(dim=-1).mean().item()
    res["illegal_move_chance"] = illegal_move_chance

    return res


def _sample_game(
    model: HookedTransformer, temp: float, probabilistic=False
) -> list[int]:
    """Soft samples a move sequence autoregressivly from the model."""
    assert temp > 0
    device = model.cfg.device
    seq = [10]
    with t.no_grad():
        # sample 9 moves plus one end game token
        for _ in range(10):
            logits: Tensor = model(t.tensor(seq, device=device))[0, -1]
            probs = t.softmax(logits / temp, dim=0)
            token = int(t.multinomial(probs, num_samples=1).item())
            seq.append(token)
    return seq


def sample_games(
    model: HookedTransformer, temp: float, num_games: int
) -> list[list[int]]:
    games: list[list[int]] = []
    for _ in range(num_games):
        games.append(_sample_game(model, temp))
    return games


# evals return True on model error
def _check_played_repeat_moves(game: list[int]) -> bool:
    """True = bad (repeated move.)"""
    clean_game = [token for token in game if token != 9]
    set_length = len(set(clean_game))
    return set_length != len(clean_game)


def _check_if_illegal_moves(game: list[int]) -> bool:
    """True = bad (illegal move.)"""
    board = Board()
    for move in game[1:-1]:
        if board.game_state == State.ONGOING:
            try:
                board.make_move(move)
            except:
                return True
        elif move == 9:
            pass
        else:
            return True
    return False


def inappropriate_end_state(game: list[int]) -> bool:
    """True = bad (game ongoing but end token occured or prior invalid move)"""
    board = Board()
    for move in game[1:]:
        if board.game_state == State.ONGOING and move == 9:
            return True
        try:
            board.make_move(move)
        except:
            return False
    return False


def _check_played_after_game_ends(game: list[int]) -> bool:
    """True = bad (game over but no end token or prior invalid move)"""
    board = Board()
    for move in game[1:]:
        if board.game_state == State.OVER and move != 9:
            return True
        try:
            board.make_move(move)
        except:
            return False
    return False


def eval_model(
    games: list[list[int]], game_evals: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[int, dict[str, bool]]]:
    """
    Returns dictionary metric -> value
    repreated_moves, played_after_game_ends, illegal_moves, inappropriate_end_state
    """
    eval_fs = [
        _check_played_repeat_moves,
        _check_played_after_game_ends,
        inappropriate_end_state,
        _check_if_illegal_moves,
    ]
    eval_counts = {func.__name__: 0.0 for func in eval_fs}
    eval_games = {
        i: {func.__name__: False for func in eval_fs} for i in range(len(games))
    }
    game_count = len(games)
    for i, game in enumerate(games):
        for func in eval_fs:
            if func(game):
                eval_games[i][func.__name__] = True
                eval_counts[func.__name__] += 1
            else:
                eval_games[i][func.__name__] = False

    for func in eval_fs:
        eval_counts[func.__name__] /= game_count

    if game_evals:
        return eval_counts, eval_games
    else:
        return eval_counts
