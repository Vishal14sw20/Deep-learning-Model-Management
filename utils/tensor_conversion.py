import numpy as np
import torch


def to_byte_tensor(_xor_diff):
    # represent each byte as an integer form 0 - 255
    int_values = [x for x in _xor_diff]
    # form a byte tensor and reshape it to the original shape
    bt = torch.ByteTensor(int_values)
    return bt


def to_tensor(b: bytes, dt, single_value=False):
    dt = dt.newbyteorder('<')
    np_array = np.frombuffer(b, dtype=dt)
    if single_value:
        return torch.tensor(np_array[0])
    else:
        return torch.tensor(np_array)