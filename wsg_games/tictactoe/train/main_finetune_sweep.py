import multiprocessing
from wsg_games.tictactoe.train.finetune import finetune_sweep_parallel
from wsg_games.tictactoe.train.create_models import *
from wsg_games.tictactoe.data import *
from wsg_games.tictactoe.data import create_hard_label_tictactoe_data


if __name__ == "__main__":
    # Set the start method to "spawn" to avoid CUDA re-initialization issues.
    multiprocessing.set_start_method("spawn", force=True)

    device = t.device("cpu")

    pretrained_project_name = "tictactoe_pretrained_reverse_rule_no_overlap_split_start_third_200k"
    finetuned_project_name = "finetune_sweep_test_parallel"
    data_folder = '/homes/55/bwilop/wsg/data/'
    experiment_folder = '/homes/55/bwilop/wsg/experiments/'

    tictactoe_data = cache_tictactoe_data_random(data_folder + 'tictactoe_data_random_STRONG_RULE_REVERSE_RULE.pkl')
    tictactoe_train_data, tictactoe_weak_finetune_data, tictactoe_val_data, tictactoe_test_data = train_test_split_tictactoe_first_two_moves_no_overlap(tictactoe_data, 42, 15, 5, 10, device, 1234)
    tictactoe_train_data = create_hard_label_tictactoe_data(tictactoe_train_data, num_samples=1)
    tictactoe_weak_finetune_data = create_hard_label_tictactoe_data(tictactoe_weak_finetune_data, num_samples=1)
    tictactoe_val_data = create_hard_label_tictactoe_data(tictactoe_val_data, num_samples=1)

    training_cfg = get_training_cfg()

    finetune_sweep_parallel(
        pretrained_project_name,
        finetuned_project_name,
        experiment_folder,
        tictactoe_weak_finetune_data,
        tictactoe_val_data,
        tictactoe_test_data,
        training_cfg,
        8
    )
