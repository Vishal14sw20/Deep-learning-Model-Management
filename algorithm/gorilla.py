import struct
import time
import numpy as np
from bitstring import BitArray
import torch

from utils.other import binary

def input(array1, array2, aggregated_parameters, action='compress'):
    shape = array1.shape
    shape_l = len(shape)

    if shape_l == 1:
        for i in range(shape[0]):
            value1 = array1[i]
            value2 = array2[i]
            gorilla_algorithm(value1, value2, aggregated_parameters)

    elif shape_l == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                value1 = array1[i, j]
                value2 = array2[i, j]
                gorilla_algorithm(value1, value2, aggregated_parameters)
    elif shape_l == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    value1 = array1[i, j, k]
                    value2 = array2[i, j, k]
                    gorilla_algorithm(value1, value2, aggregated_parameters)
    elif shape_l == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        value1 = array1[i, j, k, l]
                        value2 = array2[i, j, k, l]
                        gorilla_algorithm(value1, value2, aggregated_parameters)




def decompress_input(aggregated_parameters, array):
    shape = array.shape
    shape_l = len(shape)
    new_array = np.empty(array.shape, dtype=array.dtype)


    if shape_l == 1:
        for i in range(shape[0]):
            aggregated_parameters, new_array[i] = decompression_algorithm(aggregated_parameters, array[i])
    elif shape_l == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                aggregated_parameters, new_array[i, j] = decompression_algorithm(aggregated_parameters, array[i, j])
    elif shape_l == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    aggregated_parameters, new_array[i, j, k] = decompression_algorithm(aggregated_parameters, array[i, j, k])
    elif shape_l == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        aggregated_parameters, new_array[i, j, k, l] = decompression_algorithm(aggregated_parameters,
                                                                                      array[i, j, k, l])
    #print("--- decompress ----")

    #print(new_array)
    return aggregated_parameters, torch.tensor(new_array)



def gorilla_algorithm(value1, value2, aggregated_parameters):
    bin_value1 = binary(value1)
    bin_value2 = binary(value2)
    bin_xor = ''.join(['1' if x != y else '0' for x, y in zip(bin_value1, bin_value2)])
    if bin_xor == "0" * 32:
        # control bit for zeros is 00
        #print(32 * '0')
        aggregated_parameters.append((BitArray(bin='111')))
    else:
        leading_zeros = bin_xor.index('1')

        # trailing_zeros = len(bin_xor) - bin_xor.rindex('1') - 1
        meaningful_bits = bin_xor.lstrip('0')
        #print(leading_zeros, meaningful_bits, len(meaningful_bits))
        if leading_zeros in [0, 1]:
            control_bit = '00'+ str(leading_zeros)
            bits = control_bit + meaningful_bits

        elif leading_zeros in [2, 3]:
            control_bit = '01' + str(leading_zeros % 2)
            bits = control_bit + meaningful_bits
            # aggregated_parameters.append((BitArray(bin='11')))
        elif 4 <= leading_zeros <= 7:
            control_bit = '101'
            # only storing last two values because we will take from above 101
            leading_bits = bin(leading_zeros)[3:]
            bits = control_bit + leading_bits + meaningful_bits
        elif leading_zeros > 7:
            control_bit = '110'
            leading_bits = BitArray(uint=int(leading_zeros), length=5).b
            bits = control_bit + leading_bits + meaningful_bits
        else:
            #print(bin_xor)
            nothing = ""

        aggregated_parameters.append(BitArray(bin=bits))


def decompress_bits(param, start_bit, our_bits):
    leading_zeros_count = int(param[start_bit:our_bits].b, 2)
    leading_bits = leading_zeros_count * "0"
    meaningful_bits_len = 32 - leading_zeros_count
    meaningful_bits = param[our_bits:our_bits + meaningful_bits_len].b
    print(leading_zeros_count, meaningful_bits, len(meaningful_bits))
    number = leading_bits + meaningful_bits
    param = param[our_bits + meaningful_bits_len:]

    return param, number

def decompression_algorithm(aggregated_parameters, value):
    control_bit = aggregated_parameters[:3].b
    if control_bit == '111':
        number = 32 * "0"
        aggregated_parameters = aggregated_parameters[3:]

    # leading zeros in 0-3
    elif control_bit in ['000', '001', '010', '011']:
        aggregated_parameters, number = decompress_bits(aggregated_parameters, 0, 3)

    # leading zeros between 4-7
    elif control_bit == '101':
        aggregated_parameters, number = decompress_bits(aggregated_parameters, 2, 5)

    # leading zeros greater then 7
    elif control_bit == '110':
        aggregated_parameters, number = decompress_bits(aggregated_parameters, 3, 8)
    else:
        print(control_bit)

    bin_value1 = binary(value)
    bin_xor = ''.join(['1' if x != y else '0' for x, y in zip(bin_value1, number)])
    int_number = struct.unpack('!f', bytes.fromhex(hex(int(bin_xor, 2))[2:].zfill(8)))[0]
    return aggregated_parameters, int_number



def compress_gorilla(base_model, derived_model, new_path):
    start_time = time.time()
    aggregated_parameters = BitArray()
    for b, d in zip(base_model.named_parameters(), derived_model.named_parameters()):
        # print(b, d)
        input(b[1].detach().numpy(), d[1].detach().numpy(), aggregated_parameters)

    aggregated_parameters.tofile(open(new_path, "wb"))

    end_time = time.time()
    #print('Total time : ' + str(end_time - start_time))


def decompress_gorilla(params, base_model):
    start_time = time.time()

    new = base_model.state_dict()
    for b in base_model.named_parameters():
        params, new[b[0]] = decompress_input(params, b[1].detach().numpy())
        # new[b[0]] = input_d(b[1].detach().numpy(), d[1].detach().numpy(), params)
    #print(params)
    base_model.load_state_dict(new)
    return base_model

    end_time = time.time()