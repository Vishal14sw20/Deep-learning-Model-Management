from bitstring import BitArray
import torch
import os

# removed test from path
path = os.path.dirname(__file__)[:-4]
from algorithm.gorilla import decompress_gorilla, compress_gorilla


base_net = torch.load(path+'weights/ant_beans/fine_tune/epochs_15/vgg_ft_15.pt', map_location=torch.device('cpu'))
derived_net = torch.load(path+'weights/ant_beans/fine_tune/epochs_20/lr001/vgg_ft_20.pt', map_location=torch.device('cpu'))

base = base_net.state_dict()
derived = derived_net.state_dict()

compress_gorilla(base_net, derived_net, path+"test_weights/vgg_gorilla_ft")

#with open("/Users/vishalkumarlohana/MyProjects/Thesis/test_weights/densenet_bits", "rb") as f:
#    data = f.read()
#    # aggregated_parameters.fromfile(f)
#params = BitArray(data)

#model = decompress_gorilla(params, base_net)
#print(model)


