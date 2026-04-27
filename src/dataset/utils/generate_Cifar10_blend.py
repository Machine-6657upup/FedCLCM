import numpy as np
import os
import sys
import random
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import check, separate_data, split_data, save_file
from PIL import Image

random.seed(42)
np.random.seed(42)
GENERATOR_SEED = 42
ASR_TEST_POOL_SIZE = 100
BASE_DIR = Path(__file__).resolve().parent
TRIGGER_DIR = BASE_DIR.parents[1] / "backdoor" / "triggers"


def get_rawdata_root(dir_path):
    return globals().get("rawdata_path", os.path.join(dir_path, "rawdata"))

def get_blend_alpha():
    return float(globals().get("blend_alpha", 0.2))


def load_blend_trigger():
    trigger = Image.open(TRIGGER_DIR / "hellokitty_32.png").convert("RGB")
    trigger = np.array(trigger).transpose(2, 0, 1)
    trigger = (trigger / 255.0) * 2 - 1
    return trigger.astype(np.float32)


def add_trigger32_blend(data_dict, backdoor_rate, target_y, alpha=None):
    feature = data_dict["x"].copy()
    label = data_dict["y"].copy()
    trigger = load_blend_trigger()
    alpha = get_blend_alpha() if alpha is None else alpha

    num_img = feature.shape[0]
    id_set = list(range(num_img))
    num_poison = int(num_img * backdoor_rate)

    poison_indices = random.sample(id_set, num_poison)

    for i in poison_indices:
        feature[i] = (1 - alpha) * feature[i] + alpha * trigger
        feature[i] = np.clip(feature[i], -1.0, 1.0)
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
        train_data[i] = add_trigger32_blend(train_data[i], backdoor_rate, target_y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    # ASR is evaluated on a fixed non-target clean-test pool for every benign
    # client. This avoids empty per-client ASR sets under extreme non-IID splits.
    backdoor_source = build_shared_backdoor_source(test_data, target_y)
    for idx, _ in enumerate(test_data):
        test_dict = add_trigger32_blend(backdoor_source, 1, target_y)
        with open(test_path + str(idx) + '_backdoored.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

def build_shared_backdoor_source(test_data, target_y, max_samples=ASR_TEST_POOL_SIZE):
    features = [d["x"] for d in test_data if d["x"].shape[0] > 0]
    labels = [d["y"] for d in test_data if d["y"].shape[0] > 0]
    if not features:
        return {'x': np.empty((0, 3, 32, 32), dtype=np.float32), 'y': np.empty((0,), dtype=np.int64)}
    feature = np.concatenate(features, axis=0)
    label = np.concatenate(labels, axis=0)
    mask = (label != target_y)
    feature = feature[mask]
    label = label[mask]
    if feature.shape[0] > max_samples:
        rng = np.random.default_rng(GENERATOR_SEED)
        indices = rng.choice(feature.shape[0], size=max_samples, replace=False)
        feature = feature[indices]
        label = label[indices]
    if feature.shape[0] == 0:
        raise RuntimeError("Cannot build ASR test pool: no non-target clean test samples.")
    return {'x': feature, 'y': label}



if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    num_clients = 40
    backdoor_rate = 0.2
    adversary_num = 5
    dir_path = f"../Cifar10_dir0.5_bdoor{backdoor_rate}_nclient_{num_clients}_blend_adv{adversary_num}/"
    target_y = 0
    aux_path = dir_path + "test/"
    server_clean_path = dir_path + "test/"

    generate_dataset(dir_path, num_clients, niid, balance, partition, backdoor_rate, target_y)

