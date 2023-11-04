import torch
import os
import time


# removed test from path
path = os.path.dirname(__file__)[:-4]

from algorithm.zlib_xor import get_xor_representation, recover_from_xor_representation
base_models = ['alexnet_ft_15', 'densenet_ft_15', 'inception_ft_15', 'resnet_ft_15', 'vgg_ft_15']
derived_models = ['alexnet_ft_20', 'densenet_ft_20', 'inception_ft_20', 'resnet_ft_20', 'vgg_ft_20']




method = 'rle'
action = 'r'


for i in range(len(base_models)):
    start_time = time.time()

    base_net = torch.load(path + 'weights/ant_beans/fine_tune/epochs_15/{}.pt'.format(base_models[i]), map_location=torch.device('cpu'))
    derived_net = torch.load(path + 'weights/ant_beans/fine_tune/epochs_20/lr001/{}.pt'.format(derived_models[i]),
                             map_location=torch.device('cpu'))

    base = base_net.state_dict()
    derived = derived_net.state_dict()

    if action == 's':
        xor_repr = get_xor_representation(base, derived, method)
        torch.save(xor_repr, path + 'test_weights/{}_{}_xor.pt'.format(derived_models[i], method))
    elif action == 'r':
        diff = torch.load(path + 'test_weights/{}_{}_xor.pt'.format(derived_models[i], method))
        recovered = recover_from_xor_representation(base, diff, method)


    end_time = time.time()
    print('Total time : ' + str(end_time - start_time))


