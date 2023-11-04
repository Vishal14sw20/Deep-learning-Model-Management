import logging


import torch
import os
import time


logging.basicConfig(filename="models_log.log", level=logging.DEBUG,
    format='%(message)s')




# removed test from path
path = os.path.dirname(__file__)[:-4]

from algorithm.zlib_xor import get_xor_representation, recover_from_xor_representation
models = ['alexnet', 'resnet' ,'ViT', 'vgg']
epochs = ['9', '19']
extractions = ['fe', 'ft']
datasets = ['ant_beans', 'image_net_subset']
lr = '0.005'


methods = ['zlib', 'gzip', 'lzma', 'rle', 'delta']
action = 's'
for method in methods:
    print(method)
    logging.info(f"Method :  {method}")
    for extraction in extractions:
        logging.info(f" Extraction :  {extraction}")
        for dataset in datasets:
            logging.info(f"  Dataset :  {dataset}")
            for model in models:
                start_time = time.time()
                logging.info(f"   Model :  {model}")

                base_path = path + 'weights/{}/{}/{}_{}_{}_lr_{}.pt'.format(dataset, extraction, model, extraction,
                                                                            epochs[0],
                                                                            lr)
                derived_path = path + 'weights/{}/{}/{}_{}_{}_lr_{}.pt'.format(dataset, extraction, model, extraction,
                                                                               epochs[1], lr)
                base_net = torch.load(base_path, map_location=torch.device('cpu'))
                derived_net = torch.load(derived_path, map_location=torch.device('cpu'))
                base_file_size_bytes = os.path.getsize(base_path)
                base_file_size_mb = base_file_size_bytes / (1024 * 1024)

                base = base_net.state_dict()
                derived = derived_net.state_dict()

                if action == 's':
                    xor_repr = get_xor_representation(base, derived, method)
                    xor_model_path = path + 'test_weights/{}_{}_{}_{}_xor.pt'.format(model, dataset, extraction, method)
                    torch.save(xor_repr, xor_model_path)

                    compressed_file_size_bytes = os.path.getsize(xor_model_path)
                    compressed_file_size_mb = compressed_file_size_bytes / (1024 * 1024)

                    end_time = time.time()

                    logging.info(f"    Size of Original '{model}': {base_file_size_mb:.2f} MB")
                    logging.info(f"    Size of Compressed '{model}': {compressed_file_size_mb:.2f} MB")
                    logging.info('    Total time in Saving Model : ' + str(end_time - start_time))

                elif action == 'r':
                    xor_model_path = path + 'test_weights/{}_{}_{}_{}_xor.pt'.format(model, dataset, extraction, method)
                    diff = torch.load(xor_model_path)
                    recovered = recover_from_xor_representation(base, diff, method)

                    recovered_size_bytes = recovered.element_size() * recovered.numel()
                    recovered_size_mb = recovered_size_bytes / (1024 * 1024)

                    end_time = time.time()

                    logging.info(f"    Size of Recovered '{model}': {recovered_size_mb:.2f} MB")
                    logging.info('    Total time in Recovering Model : ' + str(end_time - start_time))



                print('Total time : ' + str(end_time - start_time))




