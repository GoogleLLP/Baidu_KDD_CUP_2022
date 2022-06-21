# -*-Encoding: utf-8 -*-
from typing import Optional, Union
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray


class Scaler(object):
    """
    Desc: Normalization utilities\n
    """

    def __init__(self, _settings: dict):
        self.mean = 0.
        self.std = 1.
        self.target = _settings["target"]

    def fit(self, data: Union[ndarray, DataFrame, Series]) -> None:
        """
        制作一个标准化器\n
        :param data: 输入
        :return
        """
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data: Union[ndarray, DataFrame, Series]) -> Union[ndarray, DataFrame, Series]:
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data: Union[ndarray, DataFrame, Series], only_target: bool = False) -> Union[ndarray, DataFrame, Series]:
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
            only_target: 是否只变换target
        Returns:
            The original data
        """
        mean = self.mean
        std = self.std
        if only_target:
            return (data * std.loc[self.target]) + mean.loc[self.target]
        else:
            return (data * std) + mean


def slice_window(data: np.ndarray, sequence_length: int, batch_size: int) -> np.ndarray:
    """
    滑动窗口\n
    Args:
        data: ndarray
        sequence_length: 序列长度
        batch_size: 大小

    Returns:
        结果
    """
    result: ndarray = np.lib.stride_tricks.sliding_window_view(data, sequence_length, axis=0)
    result = result.transpose((0, 2, 1))
    return result.reshape(result.shape[0], -1)


def get_turb_list(data: DataFrame, turb_id: int, _settings: dict) -> ((ndarray, ), (ndarray, )):
    """
    获取一个风机的数据\n
    :param data: 原始数据
    :param turb_id: 风机id
    :param _settings: 设置项
    :return: (x_train, y_train), (x_val, y_val)
    """
    turbine_id = _settings["turbine_id"]
    start_col = _settings["start_col"]
    input_len = _settings["input_len"]
    output_len = _settings["output_len"]
    in_var = _settings["in_var"]
    out_var = _settings["out_var"]
    batch_size = _settings["batch_size"]
    train_size = _settings["train_size"]
    val_size = _settings["val_size"]
    data = data[data[turbine_id] == turb_id]
    scl = Scaler(_settings)
    scl.fit(data.iloc[:, start_col:])
    data.iloc[:, start_col:] = scl.transform(data.iloc[:, start_col:])
    xy_train = data[data["Day"] <= train_size].fillna(0).iloc[:, start_col:].values
    xy_val = data[data["Day"] > train_size].fillna(0).iloc[:, start_col:].values
    x_train = slice_window(xy_train[:-output_len, 0:in_var], sequence_length=input_len, batch_size=batch_size)
    y_train = slice_window(xy_train[input_len:, -out_var:], sequence_length=output_len, batch_size=batch_size)
    if val_size > input_len + output_len:
        x_val = slice_window(xy_val[:-output_len, 0:in_var], sequence_length=input_len, batch_size=batch_size)
        y_val = slice_window(xy_val[input_len:, -out_var:], sequence_length=output_len, batch_size=batch_size)
        return (x_train, y_train), (x_val, y_val)
    else:
        return (x_train, y_train), (None, None)


def get_turb_test_list(data: DataFrame, turb_id: int, _settings: dict) -> (ndarray, Scaler):
    """
    获取一个风机的数据\n
    :param data: 原始数据
    :param turb_id: 风机id
    :param _settings: 设置项
    :return: 测试集x
    """
    turbine_id = _settings["turbine_id"]
    start_col = _settings["start_col"]
    input_len = _settings["input_len"]
    output_len = _settings["output_len"]
    in_var = _settings["in_var"]
    out_var = _settings["out_var"]
    batch_size = _settings["batch_size"]
    data = data[data[turbine_id] == turb_id]
    scl = Scaler(_settings)
    scl.fit(data.iloc[:, start_col:])
    data.iloc[:, start_col:] = scl.transform(data.iloc[:, start_col:])
    xy_test = data.fillna(0).iloc[-input_len:, start_col:].values
    x_test = slice_window(xy_test[:, 0:in_var], sequence_length=input_len, batch_size=batch_size)
    return x_test, scl
