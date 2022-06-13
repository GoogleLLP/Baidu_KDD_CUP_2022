# -*-Encoding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import load
from common import get_turb_test_list
from prepare import prep_env


physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)


def forecast(_settings: dict) -> np.ndarray:
    """
    预测结果\n
    :param _settings: 配置
    :return: 预测值
    """
    capacity = _settings["capacity"]
    output_len = _settings["output_len"]
    results = np.zeros(shape=(capacity, output_len, 1), dtype=float)
    turbine_id = _settings["turbine_id"]
    filepath = os.path.join(_settings["data_path"], _settings["filename"])
    data = pd.read_csv(filepath)
    turb_ids = data[turbine_id].unique().tolist()
    for i, turb_id in enumerate(turb_ids):
        model: MultiOutputRegressor = load(os.path.join(_settings["checkpoints"], "model_%d.pkl" % turb_id))
        data_test, scaler = get_turb_test_list(data, turb_id, _settings)
        result = model.predict(data_test)
        result = scaler.inverse_transform(result, only_target=True)
        results[i, :, 0] = result.ravel()
    return np.array(results)


if __name__ == '__main__':
    settings = prep_env()

    print(forecast(settings))
