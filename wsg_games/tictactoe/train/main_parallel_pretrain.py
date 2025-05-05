#!/usr/bin/env python3

# python3 wsg_games/tictactoe/train/main_parallel_pretrain_sweep.py
import time
import queue
import copy
import torch as t
import torch.multiprocessing as mp
from multiprocessing import Manager

from wsg_games.tictactoe.train.train import run_full_training
from wsg_games.tictactoe.train.save_load_models import (
    load_model,
    save_model,
    load_finetuned_model_get_matching_files,
)
from transformer_lens.utilities.devices import move_to_and_update_config
from wsg_games.tictactoe.game import Goal
from wsg_games.tictactoe.train.create_models import (
    get_training_cfg,
)
from wsg_games.tictactoe.data import (
    cache_tictactoe_data_random,
    train_test_split_tictactoe_first_two_moves_no_overlap,
    create_hard_label_tictactoe_data,
    move_tictactoe_data_to_device
)
