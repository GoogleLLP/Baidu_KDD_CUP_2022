# -*-Encoding: utf-8 -*-
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump
from prepare import prep_env
from common import get_turb_list


def train_and_val(_settings: dict) -> None:
    """
    训练模型\n
    :param _settings: 关键字参数
    :return: 空
    """
    turbine_id = _settings["turbine_id"]
    filepath = os.path.join(_settings["data_path"], _settings["filename"])
    data = pd.read_csv(filepath)
    turb_ids = data[turbine_id].unique().tolist()
    for turb_id in turb_ids:
        print(">>>>>>>>>>>>> turb %d training begin >>>>>>>>>>>>>>>>>" % turb_id)
        (x_train, y_train), (_, _) = get_turb_list(data, turb_id, _settings)
        model = MultiOutputRegressor(GradientBoostingRegressor(loss="huber", learning_rate=0.1, n_estimators=10, verbose=1), n_jobs=-1)
        print(x_train.shape, y_train.shape)
        model.fit(x_train, y_train)
        dump(model, os.path.join(_settings["checkpoints"], "model_%d.pkl" % turb_id))


if __name__ == "__main__":
    settings = prep_env()

    train_and_val(settings)

