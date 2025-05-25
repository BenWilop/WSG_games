import torch as t
from typing import Tuple, List, Dict, Set, Optional, Any
from tqdm import tqdm
from wsg_games.tictactoe.data import TicTacToeData

# Define WINNING_LINES_SETS globally. These represent the sets of board positions (0-8)
# that constitute a win in Tic-Tac-Toe.
WINNING_LINES_SETS: List[Set[int]] = [
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},  # Rows
    {0, 3, 6},
    {1, 4, 7},
    {2, 5, 8},  # Columns
    {0, 4, 8},
    {2, 4, 6},  # Diagonals
]


def _is_first_move_potential_bonus_blocked(
    first_move_of_player: Optional[int],
    opponent_moves: Set[int],
    bonus_eligible_lines: List[Set[int]],
) -> bool:
    """
    Checks if all potential bonus-winning lines through the player's first move are blocked by the opponent.
    A "bonus-winning line" is a line that, if completed by the player, would grant them a bonus
    because their first_move_of_player is on that line.

    Args:
        first_move_of_player: The specific first move (0-8) made by the player in the current prefix.
                              If None, it means the player hasn't made their first move in this context,
                              so it's vacuously considered "blocked" or irrelevant.
        opponent_moves: A set of moves made by the opponent.
        bonus_eligible_lines: A list of sets, where each set represents a winning line
                              eligible for the bonus.

    Returns:
        True if all bonus-eligible lines containing first_move_of_player are blocked by an opponent's move.
        False if there's at least one such line that is not blocked by any opponent's move.
        Returns True if first_move_of_player is None.
    """
    if first_move_of_player is None:
        return True  # Player hasn't made their first move in this prefix, so no potential bonus to block yet.

    relevant_lines_for_first_move = [
        line for line in bonus_eligible_lines if first_move_of_player in line
    ]

    if not relevant_lines_for_first_move:
        return True

    for line in relevant_lines_for_first_move:
        is_line_blocked_by_opponent = any(op_move in line for op_move in opponent_moves)
        if not is_line_blocked_by_opponent:
            return False

    return True


def _characterize_prefix(
    prefix_tuple: Tuple[int, ...], check_diagonal_bonus: bool
) -> Dict[str, Any]:
    """
    Analyzes a game prefix (sequence of moves) to determine its properties.
    (Logic unchanged, but critical for the overall process)
    """
    if not prefix_tuple:
        return {
            "original_prefix": prefix_tuple,
            "first_two_moves": (None, None),
            "board_state_tuple": tuple(),
            "is_win": False,
            "winner_player_idx": None,
            "is_special_win": False,
            "first_two_moves_bonus_potential_blocked": True,
        }

    board_state_tuple = tuple(sorted(list(set(prefix_tuple))))

    p1_first_move_in_prefix = prefix_tuple[0] if len(prefix_tuple) > 0 else None
    p2_first_move_in_prefix = prefix_tuple[1] if len(prefix_tuple) > 1 else None
    first_two_moves = (p1_first_move_in_prefix, p2_first_move_in_prefix)

    p1_moves = {prefix_tuple[i] for i in range(0, len(prefix_tuple), 2)}
    p2_moves = {prefix_tuple[i] for i in range(1, len(prefix_tuple), 2)}

    is_win, winner_player_idx, is_special_win = False, None, False
    bonus_eligible_lines = (
        WINNING_LINES_SETS if check_diagonal_bonus else WINNING_LINES_SETS[:6]
    )

    for line in WINNING_LINES_SETS:
        if line.issubset(p1_moves):
            is_win, winner_player_idx = True, 0
            if p1_first_move_in_prefix is not None:
                for completed_bonus_line in bonus_eligible_lines:
                    if (
                        completed_bonus_line.issubset(p1_moves)
                        and p1_first_move_in_prefix in completed_bonus_line
                    ):
                        is_special_win = True
                        break
            break

    if not is_win:
        for line in WINNING_LINES_SETS:
            if line.issubset(p2_moves):
                is_win, winner_player_idx = True, 1
                if p2_first_move_in_prefix is not None:
                    for completed_bonus_line in bonus_eligible_lines:
                        if (
                            completed_bonus_line.issubset(p2_moves)
                            and p2_first_move_in_prefix in completed_bonus_line
                        ):
                            is_special_win = True
                            break
                break

    p1_first_move_bonus_blocked = _is_first_move_potential_bonus_blocked(
        p1_first_move_in_prefix, p2_moves, bonus_eligible_lines
    )
    p2_first_move_bonus_blocked = _is_first_move_potential_bonus_blocked(
        p2_first_move_in_prefix, p1_moves, bonus_eligible_lines
    )
    first_two_moves_bonus_potential_blocked = (
        p1_first_move_bonus_blocked and p2_first_move_bonus_blocked
    )

    return {
        "original_prefix": prefix_tuple,
        "first_two_moves": first_two_moves,
        "board_state_tuple": board_state_tuple,
        "is_win": is_win,
        "winner_player_idx": winner_player_idx,
        "is_special_win": is_special_win,
        "first_two_moves_bonus_potential_blocked": first_two_moves_bonus_potential_blocked,
    }


def get_all_characteristics(
    games_data: t.Tensor, check_diagonal_bonus: bool, min_prefix_len: int = 1
) -> List[Dict[str, Any]]:
    """
    Processes all games in a dataset to extract and characterize all their prefixes.
    Uses tqdm for progress indication.
    """
    games_data_cpu = games_data.cpu()
    all_characteristics = []
    num_games = games_data_cpu.shape[0]

    # Added tqdm here as per user's code
    for i in tqdm(range(num_games), desc=" caratteristicas"):
        game_with_tokens = games_data_cpu[i].tolist()
        actual_moves: List[int] = []
        for move_val in game_with_tokens:
            if move_val == 10:
                continue
            if move_val == 9:
                break
            actual_moves.append(move_val)

        if not actual_moves and min_prefix_len > 0:
            continue
        # if not actual_moves and min_prefix_len == 0: pass # Allows empty prefix if min_prefix_len is 0

        for l in range(min_prefix_len, len(actual_moves) + 1):
            prefix_tuple = tuple(actual_moves[:l])
            char_data = _characterize_prefix(prefix_tuple, check_diagonal_bonus)
            all_characteristics.append(char_data)

    return all_characteristics


def calculate_leakage_percentage_modified(
    train_data: TicTacToeData, test_data: TicTacToeData, check_diagonal_bonus: bool
) -> float:
    """
    Computes the percentage of test subgames (prefixes) that are considered duplicates
    of training subgames based on the modified rules, using optimized lookup.
    Uses tqdm for progress indication.
    """
    # print("Getting training characteristics...")
    train_characteristics_list = get_all_characteristics(
        train_data.games_data,
        check_diagonal_bonus,
        min_prefix_len=1,  # Get all possible train prefixes (len 1 to full game)
    )

    # --- Optimization: Pre-process training characteristics for fast lookup ---
    train_cond1_fingerprints = set()
    # Stores board_state_tuple for train_chars where its own bonus potential is blocked
    train_cond2_board_states_if_blocked = set()

    # print("Processing training characteristics for lookup...")
    for tc in tqdm(train_characteristics_list, desc="Pre-process Train Chars"):
        # For Condition 1: requires the train prefix to also have at least two moves
        if tc["first_two_moves"][1] is not None:
            fingerprint = (tc["first_two_moves"], tc["board_state_tuple"])
            train_cond1_fingerprints.add(fingerprint)

        # For Condition 2: store the board state if its bonus potential is blocked
        if tc["first_two_moves_bonus_potential_blocked"]:
            train_cond2_board_states_if_blocked.add(tc["board_state_tuple"])
    # --- End of Optimization Setup ---

    leaked_count = 0
    total_test_prefixes_evaluated = 0
    leaked_count_actual_games = 0
    total_test_prefixes_evaluated_actual_games = 0

    test_games_tensor_cpu = test_data.games_data.cpu()
    num_test_games = test_games_tensor_cpu.shape[0]

    # print("Checking test games for leakage...")
    # Added tqdm here as per user's code
    for i in tqdm(range(num_test_games), desc="Checking Test Games"):
        game_with_tokens = test_games_tensor_cpu[i].tolist()
        actual_moves: List[int] = []
        for move_val in game_with_tokens:
            if move_val == 10:
                continue
            if move_val == 9:
                break
            actual_moves.append(move_val)

        if not actual_moves:
            continue

        # Iterate through test prefixes of length 3 up to full game length
        for l in range(3, len(actual_moves) + 1):
            test_prefix_tuple = tuple(actual_moves[:l])
            total_test_prefixes_evaluated_actual_games += 1
            current_test_char = _characterize_prefix(
                test_prefix_tuple, check_diagonal_bonus
            )

            found_leak_for_this_prefix = False

            # --- Optimized Leak Check ---
            # Condition 1: First two moves AND board state are the same.
            # current_test_char always has first_two_moves[1] not None because l >= 3.
            test_c1_fingerprint = (
                current_test_char["first_two_moves"],
                current_test_char["board_state_tuple"],
            )
            if test_c1_fingerprint in train_cond1_fingerprints:
                leaked_count_actual_games += 1
                found_leak_for_this_prefix = True

            # Condition 2: (Only if not leaked by Cond 1)
            # Board states are the same, AND for both test and train prefixes,
            # the bonus potential of their respective first two moves is blocked.
            if (
                not found_leak_for_this_prefix
                and current_test_char["first_two_moves_bonus_potential_blocked"]
                and current_test_char["board_state_tuple"]
                in train_cond2_board_states_if_blocked
            ):
                leaked_count_actual_games += 1
                # found_leak_for_this_prefix = True # Not strictly needed as it's the last check
            # --- End of Optimized Leak Check ---

        left_subgames = 10 - len(actual_moves)
        total_test_prefixes_evaluated += left_subgames
        leaked_count += left_subgames

    leaked_count += leaked_count_actual_games
    total_test_prefixes_evaluated += total_test_prefixes_evaluated_actual_games

    if total_test_prefixes_evaluated == 0:
        return 0.0
    leakage_percentage_actual_games = (
        leaked_count_actual_games / total_test_prefixes_evaluated_actual_games
    ) * 100
    leakage_percentage = (leaked_count / total_test_prefixes_evaluated) * 100
    return leakage_percentage_actual_games, leakage_percentage
