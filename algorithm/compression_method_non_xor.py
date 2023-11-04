import copy
import gzip
import lzma
import math
import zlib

import numpy as np
import torch

from algorithm.delta import delta_compress, delta_decompress
from algorithm.rle import rle_compress, rle_decompress
from utils.tensor_conversion import to_byte_tensor
from utils.tensor_conversion import to_tensor
from utils.type_conversion import torch_dtype_to_numpy_dict


def _bytes_per_value(_type: torch.dtype):
    # to be more efficient we can just create/save a lookup table
    dummy_tensor = torch.tensor([1], dtype=_type)
    return len(dummy_tensor.numpy().tobytes())


def save_model(model, new_path, compress_type):
    if compress_type == 'zlib':
        aggregated_parameters = bytearray()
        for _, tensor, in model.items():
            aggregated_parameters.extend(tensor.numpy().tobytes())

        compressed_params = zlib.compress(aggregated_parameters)
        # convert to a byte tensor for efficient storage
        byte_tensor = to_byte_tensor(compressed_params)
        # save byte tensor to disk
        torch.save(byte_tensor, new_path)
        print("Zlib compression Done")


def get_representation(model, compress_type):
    aggregated_parameters = bytearray()
    for _, tensor, in model.items():
        aggregated_parameters.extend(tensor.numpy().tobytes())

    if compress_type == 'zlib':
        compressed_params = zlib.compress(aggregated_parameters)
    elif compress_type == 'gzip':
        compressed_params = gzip.compress(aggregated_parameters)
    elif compress_type == 'lzma':
        compressed_params = lzma.compress(aggregated_parameters)
    elif compress_type == 'rle':
        compressed_params = rle_compress(aggregated_parameters)
    elif compress_type == 'delta':
        compressed_params = delta_compress(aggregated_parameters)

    byte_tensor = to_byte_tensor(compressed_params)

    return byte_tensor


def recover_model(base_model, data, tpye):
    #data = torch.load(new_path)
    diff = data.numpy().tobytes()
    if tpye == 'zlib':
        data = zlib.decompress(diff)
    elif tpye == 'gzip':
        data = gzip.decompress(diff)
    elif tpye == 'lzma':
        data = lzma.decompress(diff)
    elif tpye == 'rle':
        data = rle_decompress(diff)
    elif tpye == 'delta':
        data = delta_decompress(diff)
    result = []

    # position byte array to read form
    byte_pointer = 0
    model = copy.deepcopy(base_model)
    # read until we have no data left
    while byte_pointer < len(data):
        # create a copy of the example model and load new weights in it
        # model = copy.deepcopy(base_net)
        model_state = model.state_dict()
        for k, _tensor in model_state.items():
            # the number of bytes we have to extract is equivalent to:
            # the number of values the tensor holds * number of bytes for the datatype (for one float 4 bytes)
            shape = _tensor.shape
            num_bytes = math.prod(shape) * _bytes_per_value(_tensor.dtype)
            # read the bytes for the tensor
            byte_data = data[byte_pointer:byte_pointer + num_bytes]
            # form tensor out of bytes, and reshape
            np_dtype = np.dtype(torch_dtype_to_numpy_dict[_tensor.dtype])
            recovered_tensor = to_tensor(byte_data, np_dtype)
            recovered_tensor = torch.reshape(recovered_tensor, shape)
            # override the recovered tensor in the state dict
            model_state[k] = recovered_tensor

            # update byte pointer to read form correct position
            byte_pointer += num_bytes

        # as soon as state_dict for one model is recovered load it back into model and append it to result
        model.load_state_dict(model_state)
    return model

