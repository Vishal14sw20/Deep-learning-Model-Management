import time
from bitstring import BitArray
import torch

from algorithm.gorilla import decompress_gorilla

base_net = torch.load('weights/torch_squeeze_weights.pt', map_location=torch.device('cpu'))
derived_net = torch.load('weights/torch_squeeze_weights_fine_tune.pt', map_location=torch.device('cpu'))

base = base_net.state_dict()
derived = derived_net.state_dict()


with open("test_weights/squeeze_bits", "rb") as f:
    data = f.read()
    # aggregated_parameters.fromfile(f)
params = BitArray(data)

decompress_gorilla(params, base_net)

