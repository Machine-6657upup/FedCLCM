import numpy as np
import os
import sys
import random
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import check, separate_data, split_data, save_file

random.seed(42)
np.random.seed(42)
GENERATOR_SEED = 42
BASE_DIR = Path(__file__).resolve().parent


def get_rawdata_root(dir_path):
    return globals().get("rawdata_path", os.path.join(dir_path, "rawdata"))

def get_sig_delta():
    return float(globals().get("sig_delta", 30 / 255))

def get_sig_f():
    return int(globals().get("sig_f", 6))

def get_sig_label_mode():
    return globals().get("sig_label_mode", "dirty")

def build_sig_trigger(img_c=3, img_h=32, img_w=32, delta=None, f=None):
    delta = get_sig_delta() if delta is None else delta
    f = get_sig_f() if f is None else f
    pattern = np.zeros((img_h, img_w), dtype=np.float32)
    for i in range(img_h):
        for j in range(img_w):
            pattern[i, j] = delta * np.sin(2 * np.pi * j * f / img_w)

    # CIFAR10 generator normalizes images from [0,1] to [-1,1], so an additive
    # raw-space perturbation of delta should be scaled by 1/std = 2 in normalized space.
    pattern = 2.0 * pattern
    pattern = np.repeat(pattern[None, :, :], img_c, axis=0)
    return pattern

def add_trigger32_sig(data_dict, backdoor_rate, target_y, delta=None, f=None, label_mode=None):
    feature = data_dict["x"].copy()
    label = data_dict["y"].copy()
    trigger = build_sig_trigger(
        img_c=feature.shape[1],
        img_h=feature.shape[2],
        img_w=feature.shape[3],
        delta=delta,
        f=f,
    )
    label_mode = get_sig_label_mode() if label_mode is None else label_mode

    num_img = feature.shape[0]
    if label_mode == "clean":
        id_set = [idx for idx in range(num_img) if label[idx] == target_y]
    else:
        id_set = list(range(num_img))
    num_poison = int(num_img * backdoor_rate)
    num_poison = min(num_poison, len(id_set))

    poison_indices = random.sample(id_set, num_poison)

    for i in poison_indices:
        feature[i] = feature[i] + trigger
        feature[i] = np.clip(feature[i], -1.0, 1.0)
        if label_mode == "dirty":
            label[i] = target_y

    return {'x': feature, 'y': label}



# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, backdoor_rate, target_y):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
        
    # Get Cifar10 data

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    raw_root = get_rawdata_root(dir_path)
    trainset = torchvision.datasets.CIFAR10(
        root=raw_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=raw_root, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)

    for i in range(adversary_num):
        train_data[i] = add_trigger32_sig(train_data[i], backdoor_rate, target_y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    for idx, test_dict in enumerate(test_data):
        test_dict = remove_target_test_data(test_dict)
        test_dict = add_trigger32_sig(test_dict, 1, target_y)
        with open(test_path + str(idx) + '_backdoored.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

def remove_target_test_data(dict):
    feature = dict["x"]
    label = dict["y"]
    mask = (label != target_y)
    feature = feature[mask]
    label = label[mask]
    return {'x': feature, 'y': label}



if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    num_clients = 40
    backdoor_rate = 0.2
    adversary_num = 5
    dir_path = f"../Cifar10_dir0.5_bdoor{backdoor_rate}_nclient_{num_clients}_sig_adv{adversary_num}/"
    target_y = 0
    aux_path = dir_path + "test/"
    server_clean_path = dir_path + "test/"

    generate_dataset(dir_path, num_clients, niid, balance, partition, backdoor_rate, target_y)

