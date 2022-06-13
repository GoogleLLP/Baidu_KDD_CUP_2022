# -*- coding: utf-8 -*-


def prep_env() -> dict:
    """
    准备超参数\n
    :return: 保存超参数的字典
    """
    settings = {
        "checkpoints": "models",
        "pred_file": "predict.py",
        "start_col": 3,
        "framework": "tensorflow",
        "data_path": "./data",
        "filename": "wtbdata_245days.csv",
        "target": "Patv",
        "turbine_id": "TurbID",
        "input_len": 144,
        "output_len": 288,
        "in_var": 10,
        "out_var": 1,
        "train_epochs": 50,
        "num_workers": 12,
        "batch_size": 35280,
        "logdir": "./logs",
        "patience": 10,
        "train_size": 245,
        "val_size": 0,
        "test_size": 0,
        "capacity": 134,
        "lr": 1e-4,
        "is_debug": True,
        "dropout": 0.05,
        "lr_adjust": "type1",




        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "task": "MS",
        "day_len": 144,
        "total_size": 245,
        "gpu": 0,

    }

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
