import pickle as pkl
import os
import urllib.request


def show_progress(count, block_size, total_size):
    blocks = int(total_size / block_size)
    amount_finished = count / blocks
    progress = int(amount_finished * 30)
    if progress != 30:
        protgress_bar = '|' + '=' * progress + '>' + ' ' * (30 - progress - 1) + '|'
    else:
        protgress_bar = '|' + '=' * progress + '|'

    print(f"\r{count} of {blocks} blocks downloaded - {protgress_bar} - {100 * amount_finished:.2f}%", end='', flush=True)

def load_ndarrays(data_dict, path):
    for i in data_dict.keys():
        fileName = path + i
        fileObject = open(fileName, 'rb')
        data_dict[i] = pkl.load(fileObject)
        fileObject.close()


def save_ndarrays(data_dict, path):
    print("saving ndarrays")
    if not os.path.exists(path):
        os.makedirs(path)
    for i in data_dict.keys():
        fileName = path + i
        fileObject = open(fileName, 'wb')
        pkl.dump(data_dict[i], fileObject)
        fileObject.close()

def download_dataset(path, file_name, download_url):
    if not os.path.exists(path):
        print('Downloading')
        os.makedirs(path)

        urllib.request.urlretrieve(download_url,
                                   path + file_name, show_progress)
        print("\nDone downloading!")