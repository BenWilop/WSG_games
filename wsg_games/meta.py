from enum import Enum


class Game(Enum):
    TICTACTOE = 1


def game_to_ignore_first_n_moves(game: Game) -> int:
    match game:
        case Game.TICTACTOE:
            return 2
        case _:
            raise ValueError(f"Unknown game {game}")
