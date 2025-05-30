import math
from enum import Enum
from copy import deepcopy
import torch as t
from typing import Union


class State(Enum):
    OVER = 1
    ONGOING = 2


class Goal(Enum):
    WEAK_GOAL = 0  # X has 3 in a row/column/diagonal -> X wins
    STRONG_GOAL = (
        1  # X has 3 in a row/column -> X wins, but 3 in a diagonal -> X looses
    )

    def __str__(self):
        mapping = {
            Goal.WEAK_GOAL: "weak",
            Goal.STRONG_GOAL: "strong",
        }
        return mapping[self]


class Player(Enum):
    X = 0
    O = 1


class Board:
    """Contains a board state, i.e. move history. The game can still be ongoing."""

    def __init__(self, seq: list[Union[int, t.Tensor]] | None = None) -> None:
        """If seq != None, then the board is initialized with the moves in seq"""
        self.grid: list[Player | None] = [
            None
        ] * 9  # Game board, cell i has Player i in it
        self.moves_played: list[int] = []  # list of cell ids that have been played
        self.game_state: State = State.ONGOING
        self.turn: Player = Player.X  # X starts

        if seq is not None:
            if isinstance(seq, t.Tensor):
                seq = seq.tolist()
            for move in seq:  # type: ignore
                if move in range(9):
                    self.make_move(move)

    def _swap_turn(self) -> None:
        match self.turn:
            case Player.X:
                self.turn = Player.O
            case Player.O:
                self.turn = Player.X
            case _:
                raise ValueError(f"Unexpected turn of {self.turn}")

    def get_termination_conditions(
        self,
    ) -> tuple[list[tuple[int]], list[str], Player | None]:
        termination_conditions_field_tuples = []
        termination_conditions = []
        terminating_player = None
        win_conditions = {
            (0, 1, 2): "top row",
            (3, 4, 5): "middle row",
            (6, 7, 8): "bottom row",
            (0, 3, 6): "left column",
            (1, 4, 7): "middle column",
            (2, 5, 8): "right column",
            (0, 4, 8): "top left -> bottom right",
            (2, 4, 6): "bottom left -> top right",
        }
        for condition in win_conditions.keys():
            if (self.grid[condition[0]] is not None) and (
                self.grid[condition[0]]
                == self.grid[condition[1]]
                == self.grid[condition[2]]
            ):
                terminating_player = self.grid[condition[0]]
                termination_conditions_field_tuples.append(condition)
                termination_conditions.append(win_conditions[condition])

        return (
            termination_conditions_field_tuples,
            termination_conditions,
            terminating_player,
        )

    def _set_game_state(self) -> None:
        _, termination_conditions, _ = self.get_termination_conditions()
        if termination_conditions or all(cell is not None for cell in self.grid):
            self.game_state = State.OVER
        else:
            self.game_state = State.ONGOING

    def get_winner(self, goal: Goal) -> tuple[Player | None, bool]:
        """
        Returns winner and bool indicating if the winning player
        finished 3 with the first placed stone in it.
        """
        assert self.game_state == State.OVER, "Game is not over yet."
        assert len(self.moves_played) >= 2, "Only 2 moves played so far."
        (
            termination_conditions_field_tuples,
            termination_conditions,
            terminating_player,
        ) = self.get_termination_conditions()
        if terminating_player is Player.X:
            first_move = self.moves_played[0]
        elif terminating_player is Player.O:
            first_move = self.moves_played[1]
        else:
            first_move = -1234  # Impossible move
        match goal:
            case Goal.WEAK_GOAL:
                if any(
                    [
                        first_move in cond_tuple
                        for cond_tuple in termination_conditions_field_tuples
                    ]
                ):
                    return terminating_player, True
                else:
                    return terminating_player, False
            case Goal.STRONG_GOAL:
                ################################
                #         No diagonals         #
                ################################
                if (
                    "top left -> bottom right" in termination_conditions
                    or "bottom left -> top right" in termination_conditions
                ):
                    match terminating_player:
                        case Player.X:
                            return Player.O, False
                        case Player.O:
                            return Player.X, False
                        case _:
                            raise ValueError(
                                f"Unexpected terminating_player {terminating_player}. None is forbidden here as well."
                            )
                else:
                    if any(
                        [
                            first_move in cond_tuple
                            for cond_tuple in termination_conditions_field_tuples
                        ]
                    ):
                        return terminating_player, True
                    else:
                        return terminating_player, False
            case _:
                raise ValueError(f"Unexpected goal {goal}")

    def get_possible_moves(self) -> list[int]:
        match self.game_state:
            case State.OVER:
                return []
            case State.ONGOING:
                return [i for i in range(9) if self.grid[i] == None]
            case _:
                raise ValueError(f"Unexpected game state of {self.game_state}")

    def make_move(self, move: int) -> None:
        if move not in self.get_possible_moves():
            raise ValueError(f"{move} is not a valid move.")
        self.grid[move] = self.turn
        self.moves_played.append(move)
        self._set_game_state()
        self._swap_turn()
        return self

    def undo(self) -> None:
        last_move = self.moves_played.pop()
        self.grid[last_move] = None
        self._set_game_state()
        self._swap_turn()

    def __str__(self) -> str:
        def cell_to_str(cell):
            if cell is None:
                return " "
            elif cell == Player.X:
                return "X"
            elif cell == Player.O:
                return "O"
            else:
                return str(cell)

        row1 = "| {} | {} | {} |".format(
            cell_to_str(self.grid[0]),
            cell_to_str(self.grid[1]),
            cell_to_str(self.grid[2]),
        )
        row2 = "| {} | {} | {} |".format(
            cell_to_str(self.grid[3]),
            cell_to_str(self.grid[4]),
            cell_to_str(self.grid[5]),
        )
        row3 = "| {} | {} | {} |".format(
            cell_to_str(self.grid[6]),
            cell_to_str(self.grid[7]),
            cell_to_str(self.grid[8]),
        )

        return row1 + "\n" + row2 + "\n" + row3

    def replay(self) -> None:
        temp_board = Board()
        for i, move in enumerate(self.moves_played):
            temp_board.make_move(move)
            print(f"\nAfter move {i} (move: {move}):")
            print(temp_board)


def generate_all_games(
    boards: list[Board] | None = None, finished_boards: list[Board] | None = None
) -> list[Board]:
    if boards is None:
        boards = [Board()]
    if finished_boards is None:
        print("Generating all games...")
        finished_boards = []
    ongoing_boards: list[Board] = []
    for board in boards:
        possible_moves = board.get_possible_moves()
        if possible_moves != []:
            for move in possible_moves:
                _board = deepcopy(board)
                ongoing_boards.append(_board.make_move(move))
        else:
            finished_boards.append(board)

    if ongoing_boards == []:
        print("Finished generating all games")
        return finished_boards
    else:
        return generate_all_games(ongoing_boards, finished_boards=finished_boards)


def minimax(board: Board, goal: Goal) -> int:
    """
    Player X is the maximizer and Player O is the minimizer
    If a player wins, he gets 1 point. If he completed a row of 3 with his first stone
    in it, he gets 2.
    A draw gives 0.
    """
    if board.game_state == State.OVER:
        winner, finished_with_first = board.get_winner(goal)
        match winner:
            case None:
                return 0  # draw
            case Player.X:
                if finished_with_first:
                    return 2
                else:
                    return 1
            case Player.O:
                if finished_with_first:
                    return -2
                else:
                    return -1
            case _:
                raise ValueError(f"Unexpected winner in minimax {winner}")
    else:  # recursion
        scores: list[int] = []
        for move in board.get_possible_moves():
            board.make_move(move)
            scores.append(minimax(board, goal))
            board.undo()

    # Max if X's turn, Min if O's turn
    match board.turn:
        case Player.X:
            return max(scores)
        case Player.O:
            return min(scores)
        case _:
            raise ValueError(f"Unexpected turn in minimax {board.turn}")


def get_best_moves(board: Board, goal: Goal) -> list[int]:
    """Player X is the maximizer and Player O is the minimizer"""
    match board.turn:
        case Player.X:
            best_score = -math.inf
        case Player.O:
            best_score = math.inf
        case _:
            raise ValueError(f"Unexpected turn in get_best_moves {board.turn}")

    best_moves_and_score: list[tuple[int, int]] = []  # (move, score)
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board, goal)
        board.undo()
        match board.turn:
            case Player.X:
                if score >= best_score:
                    best_score = score
                    best_moves_and_score.append((move, score))
            case Player.O:
                if score <= best_score:
                    best_score = score
                    best_moves_and_score.append((move, score))
            case _:
                raise ValueError(
                    f"Unexpected turn in get_best_moves for loop {board.turn}"
                )

    assert best_moves_and_score != []
    best_moves = [
        move for move, score in best_moves_and_score if score == best_score
    ]  # extract moves
    return best_moves
