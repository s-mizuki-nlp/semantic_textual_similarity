#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Tuple, Dict
import numpy as np
from h5py._hl.group import Group
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection

def _randomize_numpy_array(arr: np.ndarray, method: str, **kwargs) -> np.ndarray:

    _AVAILABLE_METHODS = "permute,gaussian,xavier,he"

    assert method in _AVAILABLE_METHODS.split(","), f"method must be one of those: {_AVAILABLE_METHODS}"

    if method == "permute":
        return np.random.permutation(arr)
    elif method == "gaussian":
        shape = arr.shape
        size = arr.size
        mu, std = np.mean(arr), np.std(arr)
        ret = np.random.normal(loc=mu, scale=std, size=size).reshape(shape)
        return ret
    elif method in ("xavier","he"):
        shape = arr.shape
        size = arr.size
        input_size, output_size = kwargs["input_size"], kwargs["output_size"]
        if method == "xavier":
            std = np.sqrt(6. / (input_size + output_size))
        elif method == "he":
            std = np.sqrt(6. / input_size)
        else:
            raise NotImplementedError(f"unknown method: {method}")
        ret = np.random.uniform(-std, std, size=size).reshape(shape)
        return ret
    else:
        raise NotImplementedError(f"method must be one of those: {_AVAILABLE_METHODS}")

def initialize_lstm_params(lstm: LstmCellWithProjection) -> Dict[str, np.ndarray]:
    lstm.reset_parameters()
    w_0, b, w_p_0 = extract_lstm_params_with_serialized_order(lstm)
    dict_ret = {
        "W_0":w_0,
        "B":b,
        "W_P_0":w_p_0
    }
    return dict_ret

def randomize_lstm_params(lstm_weight_group: Group, method: str, process_per_cell: bool = True) -> Dict[str, np.ndarray]:
    dict_ret = {}
    lst_weight_names = "W_0,B,W_P_0".split(",")

    if not process_per_cell:
        for weight_name in lst_weight_names:
            dict_ret[weight_name] = _randomize_numpy_array(arr=lstm_weight_group[weight_name].value, method=method)
        return dict_ret

    # else
    n_cell = 4
    # w_0_: (2*io_size, n_cell*cell_size)
    # b_: (n_cell*cell_size,)
    # w_p_0_: (cell_size, io_size)
    w_0_, b_, w_p_0_ = [lstm_weight_group[name].value for name in lst_weight_names]
    cell_size = b_.size // n_cell
    io_size = w_p_0_.shape[1]
    assert w_0_.shape[1] == b_.size, "size mismatch detected."
    assert w_p_0_.shape[0] == cell_size, "size mismatch detected."
    assert w_0_.shape[0] == 2*io_size, "size mismatch detected."
    # split W_0 into two parts so that we can distinguish the input transformation and state transformation.
    w_0_left_, w_0_right_ = w_0_[:io_size, :], w_0_[io_size:, :]

    # 1. randomize W_0 and B
    w_0_left = np.zeros_like(w_0_left_)
    w_0_right = np.zeros_like(w_0_right_)
    b = np.zeros_like(b_)
    for cell_idx in range(n_cell):
        cell_slice = slice(cell_idx*cell_size, (cell_idx+1)*cell_size)
        w_0_left[:, cell_slice] = _randomize_numpy_array(arr=w_0_left_[:, cell_slice],
                                                         method=method, input_size=io_size, output_size=cell_size)
        w_0_right[:, cell_slice] = _randomize_numpy_array(arr=w_0_right_[:, cell_slice], method=method,
                                                          input_size=io_size, output_size=cell_size)
        b[cell_slice] = _randomize_numpy_array(arr=b_[cell_slice], method=method, input_size=io_size, output_size=cell_size)
    w_0 = np.vstack([w_0_left, w_0_right])

    # 2. randomize W_P_0
    w_p_0 = _randomize_numpy_array(arr=w_p_0_, method=method, input_size=cell_size, output_size=io_size)

    dict_ret["W_0"] = w_0
    dict_ret["B"] = b
    dict_ret["W_P_0"] = w_p_0

    return dict_ret


def _reorder_gate_params(lstm_gate_params: np.ndarray, cell_size: int):

    cp = lstm_gate_params.copy()

    src_gate_indices = [1,2]
    tgt_gate_indices = [2,1]
    for src_gate_idx, tgt_gate_idx in zip(src_gate_indices, tgt_gate_indices):
        src_params = slice(src_gate_idx*cell_size, (src_gate_idx+1)*cell_size)
        tgt_params = slice(tgt_gate_idx*cell_size, (tgt_gate_idx+1)*cell_size)
        if lstm_gate_params.ndim == 1:
            lstm_gate_params[src_params] = cp[tgt_params]
        else:
            lstm_gate_params[:, src_params] = cp[:, tgt_params]

    return lstm_gate_params


def extract_lstm_params_with_serialized_order(lstm: LstmCellWithProjection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    w_0_left = lstm.input_linearity.weight.data.numpy().copy()
    w_0_right = lstm.state_linearity.weight.data.numpy().copy()
    cell_size = w_0_left.shape[0] // 4

    w_0 = _reorder_gate_params(np.transpose(np.hstack([w_0_left, w_0_right])), cell_size=cell_size)
    b = _reorder_gate_params(lstm.state_linearity.bias.data.numpy().copy(), cell_size=cell_size)
    w_p_0 = np.transpose(lstm.state_projection.weight.data.numpy().copy())

    # adjust forget gate bias: subtract 1.0
    forget_gate_index = 2
    s = slice(forget_gate_index*cell_size, (forget_gate_index+1)*cell_size)
    b[s] = b[s] - 1.0

    return (w_0, b, w_p_0)

def compare(arr1, arr2):
    assert arr1.shape == arr2.shape, "shape mismatch."
    diff = np.sum(np.abs(arr1 - arr2)) / arr1.size
    return diff

def compare_lstm_params_and_weights(lstm: LstmCellWithProjection, weight: Group):
    lst_weight_names = "W_0,B,W_P_0".split(",")

    lst_weights_gt = [weight[name].value for name in lst_weight_names]
    lst_weights_test = extract_lstm_params_with_serialized_order(lstm)

    for weight_name, weight_gt, weight_test in zip(lst_weight_names, lst_weights_gt, lst_weights_test):
        diff = compare(weight_gt, weight_test)
        print(f"object: {weight_name}, diff:{diff:3g}")