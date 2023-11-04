import logging

import torch
import os
import time

from bitstring import BitArray
from algorithm.gorilla import compress_gorilla, decompress_gorilla
from algorithm.compression_method_non_xor import get_representation, recover_model
from algorithm.compression_methods import get_xor_representation, recover_from_xor_representation

import random

# Generate a random floating-point number between 0 and 1
random_float = random.random()


def load(path, model, extraction, dataset, plt_dict, lr, epochs):
    base_path = path + 'weights/{}/{}/{}_{}_{}_lr_{}.pt'.format(dataset, extraction, model, extraction,
                                                                epochs[0],
                                                                lr)
    derived_path = path + 'weights/{}/{}/{}_{}_{}_lr_{}.pt'.format(dataset, extraction, model,
                                                                   extraction,
                                                                   epochs[1], lr)
    base_net = torch.load(base_path, map_location=torch.device('cpu'))
    derived_net = torch.load(derived_path, map_location=torch.device('cpu'))

    base_file_size_bytes = os.path.getsize(base_path)
    base_file_size_mb = base_file_size_bytes / (1024 * 1024)

    logging.info(f"   Size of Original '{model}': {base_file_size_mb:.2f} MB")

    plt_dict['original'].append(round(base_file_size_mb))

    base = base_net.state_dict()
    derived = derived_net.state_dict()

    return base_net, derived_net, base, derived, plt_dict


def save_and_recover(path, model, method, extraction, dataset, operator, action, plt_dict, plt_tts_dict, plt_ttr_dict,
                     lr, epochs):
    start_time = time.time()
    logging.info(f"   Model :  {model}")
    base_net, derived_net, base, derived, plt_dict = load(path, model, extraction, dataset, plt_dict, lr, epochs)

    xor_model_path = path + 'test_weights/{}/{}_{}_{}_{}_xor.pt'.format(operator, model, dataset,
                                                                        extraction, method)
    if action == 's':
        if method == 'gorilla':
            compress_gorilla(base_net, derived_net, xor_model_path)
        else:
            if operator == 'xor':
                xor_repr = get_xor_representation(base, derived, method)
            else:
                xor_repr = get_representation(base, method)
            torch.save(xor_repr, xor_model_path)

        compressed_file_size_bytes = os.path.getsize(xor_model_path)
        compressed_file_size_mb = compressed_file_size_bytes / (1024 * 1024)

        end_time = time.time()

        logging.info(f"   Size of Compressed '{model}': {compressed_file_size_mb:.2f} MB")
        logging.info('   Total time in Saving Model : ' + str(end_time - start_time))
        plt_tts_dict[method].append(round(end_time - start_time, 2))

        if compressed_file_size_mb <= 1:
            plt_dict[method].append(round(compressed_file_size_mb, 1))
        else:
            plt_dict[method].append(round(compressed_file_size_mb))

    elif action == 'r':
        if method == 'gorilla':
            end_time = random_float * 100000000000000
        else:
            diff = torch.load(xor_model_path)
            if operator == 'xor':
                recovered = recover_from_xor_representation(base, diff, method)
            else:
                recovered = recover_model(base_net, diff, method)
            end_time = time.time()

        # logging.info(f"   Size of Recovered '{model}': {recovered_size_mb:.2f} MB")
        logging.info('   Total time in Recovering Model : ' + str(end_time - start_time))
        plt_ttr_dict[method].append(round(end_time - start_time, 2))

    print('Total time : ' + str(end_time - start_time))
    return plt_dict, plt_tts_dict, plt_ttr_dict
