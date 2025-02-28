import math
from typing import Optional
from enum import Enum
from copy import deepcopy


class State(Enum):
    OVER = 1
    ONGOING = 2

class Goal(Enum):
    WEAK_GOAL = 0   # X wins if X has 3 in a row
    STRONG_GOAL = 1  # X wins if O has 3 in a row

class Player(Enum):
    X = 0
    O = 1

class Board:
    def __init__(self) -> None:
        self.grid: list[Player | None] = [None] * 9  # Game board, cell i has Player i in it
        self.moves_played: list[int] = []  # list of cell ids that have been played
        self.game_state: State = State.ONGOING
        self.turn: Player = Player.X  # X starts

    def _swap_turn(self) -> None:
        match self.turn:
            case Player.X:
                self.turn = Player.O
            case Player.O:
                self.turn = Player.X
            case _:
                raise ValueError(f"Unexpected turn of {self.turn}")
            
    def get_termination_conditions(self) -> tuple[list[str], Player | None]:
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
            if (
                self.grid[condition[0]]
                == self.grid[condition[1]]
                == self.grid[condition[2]]
                != None
            ):
                terminating_player = self.grid[condition[0]]
                termination_conditions.append(win_conditions[condition])

        return termination_conditions, terminating_player
            
    def _set_game_state(self) -> State:
        termination_conditions, _ = self.get_termination_conditions()
        if termination_conditions or all(cell is not None for cell in self.grid):
            self.game_state = State.OVER
        else:
            self.game_state = State.ONGOING
    
    def get_winner(self, goal: Goal) -> Player | None:
        _, terminating_player = self.get_termination_conditions()
        match goal:
            case Goal.WEAK_GOAL:
                return terminating_player
            case Goal.STRONG_GOAL:
                match terminating_player:
                    case None:
                        return None
                    case Player.X:
                        return Player.O
                    case Player.O:
                        return Player.X
                    case _:
                        raise ValueError(f"Unexpected terminating_player {terminating_player}")
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
        row1 = "| {} | {} | {} |".format(self.grid[0], self.grid[1], self.grid[2])
        row2 = "| {} | {} | {} |".format(self.grid[3], self.grid[4], self.grid[5])
        row3 = "| {} | {} | {} |".format(self.grid[6], self.grid[7], self.grid[8])
        return row1 + '\n' + row2 + '\n' + row3

def generate_all_games(
    boards: list[Board] = [Board()], finished_boards: Optional[list[Board]] = None
) -> list[Board]:
    if finished_boards == None:
        print('Generating all games...')
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
        print('Finished generating all games')
        return finished_boards
    else:
        return generate_all_games(ongoing_boards, finished_boards=finished_boards)

def minimax(board: Board, goal: Goal) -> int:
    """Player X is the maximizer and Player O is the minimizer"""
    if board.game_state == State.OVER:
        winner = board.get_winner(goal)
        match winner:
            case None:
                return 0  # draw
            case Player.X:
                return 1
            case Player.O:
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

    best_moves: list[tuple[int, int]] = []  # (move, score)
    for move in board.get_possible_moves():
        board.make_move(move)
        score = minimax(board, goal)
        board.undo()
        match board.turn:
            case Player.X:
                if score >= best_score:
                    best_score = score
                    best_moves.append((move, score))
            case Player.O:
                if score < best_score:
                    best_score = score
                    best_moves.append((move, score))
            case _:
                raise ValueError(f"Unexpected turn in get_best_moves for loop {board.turn}")

    assert best_moves != []
    best_moves = [move for move, score in best_moves if score == best_score]  # extract moves
    return best_moves
