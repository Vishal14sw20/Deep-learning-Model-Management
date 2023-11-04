import struct
import time
from unicodedata import decimal

import numpy as np
from bitstring import BitArray
from sys import getsizeof
import torch
import pickle

# Save the list of bitarray objects to a file
def save_list_to_file(bitarray_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(bitarray_list, f)


def binary(num):
    # return ('{:0>8b}'.format(c) for c in struct.pack('!f', num))
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))