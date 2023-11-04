import zlib
from collections import OrderedDict
import gzip
import lzma

import numpy as np
import torch

from utils.type_conversion import torch_dtype_to_numpy_dict
from utils.tensor_conversion import to_byte_tensor, to_tensor
from algorithm.rle import rle_compress, rle_decompress
from algorithm.delta import delta_compress, delta_decompress

DTYPE = 'dtype'
SHAPE = 'shape'
XOR_DIFF = 'xor_diff'


def get_xor_representation(s_dict1: OrderedDict, s_dict2: OrderedDict, compression: str):
    """
    Calculates and returns a xor diff between two pytorch model state dicts aka OrderedDicts.
    result = compressed(s_dict1 XOR s_dict2), so to recover s_dict2 the following can be performed: s_dict1 XOR uncompress(result)
    :param compression: algorithm for compression
    :param s_dict1: the base state dict
    :param s_dict2: the state dict to encode using XOR
    :return: compressed(s_dict1 XOR s_dict2)
    """
    # check that state dicts have the same keys meaning they are compatible
    assert s_dict1.keys() == s_dict2.keys()

    # the result will be a state dict with the same keys, but all tensors will be Bytetensors
    result = {}

    # calc the xor diff for every key
    for k in s_dict1.keys():
        t1 = s_dict1[k]
        t2 = s_dict2[k]

        # make sure the shapes of both tensors are the same meaning that the state dicts are compatible
        assert t1.shape == t2.shape
        assert t1.dtype == t2.dtype

        # calc the xor diff between two tensors and get the result as bytes
        xor_diff = _xor_diff(t1, t2)
        # compress the result
        if compression == 'zlib':
            comp_xor_diff = zlib.compress(xor_diff)
        elif compression == 'gzip':
            comp_xor_diff = gzip.compress(xor_diff)
        elif compression == 'lzma':
            comp_xor_diff = lzma.compress(xor_diff)
        elif compression == 'rle':
            comp_xor_diff = rle_compress(xor_diff)
        elif compression == 'delta':
            comp_xor_diff = delta_compress(xor_diff)
        # transform byte representation to pytorch tensor
        byte_tensor_xor_diff = to_byte_tensor(comp_xor_diff)
        # save xor encoded data together with type and shape info that is necessary for decoding
        result[k] = {XOR_DIFF: byte_tensor_xor_diff, SHAPE: t1.shape, DTYPE: t1.dtype}

    return result


def recover_from_xor_representation(base: OrderedDict, diff, compression):
    """
    Recovers a model state dict aka OrederedDict form the base models state dict and the xor diff created using get_xor_representation
    :param base: The state dict of the base model
    :param diff: The xor diff created using get_xor_representation
    :return: The originally encoded state dict.
    """
    assert base.keys() == diff.keys()

    # create ordered dict as an empty state dict
    result = OrderedDict()
    for k in base.keys():
        base_tensor = base[k]
        diff_tensor = diff[k][XOR_DIFF]
        diff_shape = diff[k][SHAPE]
        diff_type = diff[k][DTYPE]

        # check if datatype and shape are matching
        assert base_tensor.dtype == diff_type, f"base tensor type is: {base_tensor.dtype} but diff tensor type is {diff_type}"
        assert base_tensor.shape == diff_shape, f"base tensor shape is: {base_tensor.shape} but diff tensor type is {diff_shape}"

        # recover the tensor from the base tensor and the xor encoded diff
        result[k] = _recover_tensor_from_diff(base_tensor, diff_tensor, shape=diff_shape, dtype=diff_type, compression=compression)

    return result


def _recover_tensor_from_diff(t: torch.tensor, diff: torch.tensor, shape, dtype: torch.dtype, compression: str):
    # get bytes of both tensors
    t1_bytes = t.numpy().tobytes()
    diff = diff.numpy().tobytes()

    # decompress it again
    if compression == 'zlib':
        deflate_diff = zlib.decompress(diff)
    elif compression == 'gzip':
        deflate_diff = gzip.decompress(diff)
    elif compression == 'lzma':
        deflate_diff = lzma.decompress(diff)
    elif compression == 'rle':
        deflate_diff = rle_decompress(diff)
    elif compression == 'delta':
        deflate_diff = delta_decompress(diff)

    # apply xor to get original data
    _bytes = _bytewise_xor(t1_bytes, deflate_diff)

    # define numpy data type
    np_dtype = np.dtype(torch_dtype_to_numpy_dict[dtype])
    # transform bytes back into tensor, reshape if necessary
    if shape == 0:
        return to_tensor(_bytes, np_dtype, single_value=True)
    else:
        result = to_tensor(_bytes, np_dtype)
        return torch.reshape(result, shape)


def _xor_diff(t1: torch.tensor, t2: torch.tensor):
    t1_bytes = t1.numpy().tobytes()
    t2_bytes = t2.numpy().tobytes()

    _bytes = _bytewise_xor(t1_bytes, t2_bytes)
    return _bytes


def _bytewise_xor(bytes1: bytes, bytes2: bytes):
    return bytes([a ^ b for a, b in zip(bytes1, bytes2)])
