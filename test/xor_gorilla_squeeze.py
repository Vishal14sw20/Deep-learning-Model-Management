import time
import torch
import os
from algorithm.gorilla import compress_gorilla
path = os.path.dirname(__file__)[:-4]


base_net = torch.load(path+'weights/torch_squeeze_weights.pt', map_location=torch.device('cpu'))
#base_net = torch.load('th', map_location=torch.device('cpu'))
derived_net = torch.load(path+'weights/torch_squeeze_weights_fine_tune.pt', map_location=torch.device('cpu'))

base = base_net.state_dict()
derived = derived_net.state_dict()

base_layer1 = base['features.0.weight'].reshape(96, 147)
derived_layer1 = derived['features.0.weight'].reshape(96, 147)


compress_gorilla(base_net, derived_net, path+"test_weights/squeeze_bits2.pt")





