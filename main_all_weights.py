import logging

import torch
import os
import time

from utils.models_v2 import save_and_recover
from utils.create_plots import size_plot, size_plot2
from bitstring import BitArray
from algorithm.gorilla import compress_gorilla, decompress_gorilla
from algorithm.compression_method_non_xor import get_representation, recover_model
from algorithm.compression_methods import get_xor_representation, recover_from_xor_representation

# removed test from path
path = os.path.dirname(__file__) + '/'

logging.basicConfig(filename=path + "models_log.log", level=logging.DEBUG,
                    format='%(message)s')

models = ['resnet', 'ViT', 'vgg']
epochs = ['9', '19']
extractions = ['ft', 'fe']
datasets = ['ant_beans', 'image_net_subset']
lr = '0.005'
iteration = 5
plt_dict = {}
plt_tts_dict = {}
plt_ttr_dict = {}
all_dict = {}

methods = ['original', 'zlib', 'gzip', 'lzma', 'delta', 'rle', 'gorilla']
action = 's'
operators = ['xor', 'non-xor']
for operator in operators:
    for dataset in datasets:
        print(dataset)
        logging.info(f"Dataset :  {dataset}")
        for extraction in extractions:
            print(extraction)
            logging.info(f" Extraction :  {extraction}")
            for method in methods:
                print(method)
                logging.info(f"  Method :  {method}")
                size_sum = [0 for i in range(len(models))]
                tts_sum = [0 for i in range(len(models))]
                ttr_sum = [0 for i in range(len(models))]
                for v in range(iteration):
                    #plt_dict['original'] = []
                    plt_dict[method] = []
                    plt_tts_dict[method] = []
                    plt_ttr_dict[method] = []
                    for model in models:
                        plt_dict, plt_tts_dict, plt_ttr_dict = save_and_recover(path, model, method,
                                                                                             extraction, dataset, operator,
                                                                                             action, plt_dict, plt_tts_dict,
                                                                                             plt_ttr_dict, lr, epochs)
                    size_sum = [x + y for x, y in zip(plt_dict[method], size_sum)]
                    tts_sum = [x + y for x, y in zip(plt_tts_dict[method], tts_sum)]
                    ttr_sum = [x + y for x, y in zip(plt_ttr_dict[method], ttr_sum)]
                plt_dict[method] = [round(x / iteration, 2) for x in size_sum]
                plt_tts_dict[method] = [round(x / iteration, 2) for x in tts_sum]
                plt_ttr_dict[method] = [round(x / iteration, 2) for x in ttr_sum]

                if action == 's':
                    print(plt_dict)
                    print(plt_tts_dict)
            if action == 's':
                size_plot2(path, operator, dataset, extraction, models, plt_dict, 'Size')
                size_plot2(path, operator, dataset, extraction, models, plt_tts_dict, 'TTS')
                all_dict["{}_{}_size".format(dataset, extraction)] = plt_dict
                all_dict["{}_{}_tts".format(dataset, extraction)] = plt_tts_dict
                plt_dict = {}
                plt_tts_dict = {}
            if action == 'r':
                size_plot(path, operator, dataset, extraction, models, plt_ttr_dict, 'TTR')
                all_dict["{}_{}_ttr".format(dataset, extraction)] = plt_ttr_dict
                plt_ttr_dict = {}

print(all_dict)
