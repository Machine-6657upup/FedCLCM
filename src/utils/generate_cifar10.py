import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Add dataset/utils to python path to import dataset_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_utils_path = os.path.join(current_dir, '../dataset/utils')
sys.path.append(dataset_utils_path)

from dataset_utils import check, separate_data, split_data, save_file

random.seed(42)
np.random.seed(42)

def save_img(image_array, name):
    image_array = (image_array + 1) * 127.5
    image_array = image_array.clip(0, 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    Image.fromarray(image_array).save(name)

def add_trigger32_badnet(data_dict, backdoor_rate, target_y):
    feature = data_dict["x"]
    label = data_dict["y"]
    
    # Adjust path to point to backdoor/triggers from utils/
    trigger_path = os.path.join(current_dir, "../backdoor/triggers/badnet_patch_32.png")
    mask_path = os.path.join(current_dir, "../backdoor/triggers/mask_badnet_patch_32.png")
    
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
        
    trigger = Image.open(trigger_path)
    trigger = np.array(trigger).transpose(2, 0, 1)
    trigger = (trigger / 255.0) * 2 - 1
    
    mask = Image.open(mask_path)
    mask = np.array(mask).transpose(2, 0, 1)
    mask = mask / 255.0

    num_img = feature.shape[0]
    id_set = list(range(0, num_img))
    num_poison = int(num_img * backdoor_rate)
    
    if num_poison > 0:
        poison_indices = random.sample(id_set, num_poison)
        
        for n, i in enumerate(poison_indices):
            feature[i] = feature[i] + mask * (trigger - feature[i])
            label[i] = target_y

    return {'x': feature, 'y': label}

def remove_target_test_data(data_dict, target_y):
    feature = data_dict["x"]
    label = data_dict["y"]
    mask = (label != target_y)
    feature = feature[mask]
    label = label[mask]
    return {'x': feature, 'y': label}

def generate_dataset(dir_path, rawdata_path, num_clients, niid, balance, partition, backdoor_rate, target_y, adversary_num):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        print("Dataset already exists.")
        return

    # Get Cifar10 data
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)

    trainset = torchvision.datasets.CIFAR10(root=rawdata_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=rawdata_path, train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

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
                                    niid, balance, partition, class_per_client=10) # Set class_per_client to 10 for IID/default
    
    train_data, test_data = split_data(X, y)

    # Add backdoors to train data for adversaries
    # First 'adversary_num' clients are malicious
    for i in range(min(adversary_num, num_clients)):
        train_data[i] = add_trigger32_badnet(train_data[i], backdoor_rate, target_y)
        
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)

    # Generate backdoored test set
    for idx, test_dict in enumerate(test_data):
        test_dict = remove_target_test_data(test_dict, target_y)
        test_dict = add_trigger32_badnet(test_dict, 1, target_y)
        with open(os.path.join(test_path, f'{idx}_backdoored.npz'), 'wb') as f:
            np.savez_compressed(f, data=test_dict)

if __name__ == "__main__":
    # Default settings for FedCLCM quick test
    dir_path = os.path.join(current_dir, "../dataset/Cifar10/")
    rawdata_path = os.path.join(current_dir, "../dataset/rawdata/")
    
    num_clients = 100  # Default to 100
    # But wait, run_fedclcm_quick_test.sh sets num_clients=20
    # If I generate 100, can I use 20? 
    # The server/client code reads by ID. If I generate 100, clients 0-19 exist.
    # So generating 100 is safer.
    
    niid = False
    balance = True
    partition = 'dir'
    backdoor_rate = 0.5
    target_y = 0
    adversary_num = 10 # Matches run_fedclcm.sh (10) and covers quick_test (2)

    print("Generating Cifar10 dataset...")
    generate_dataset(dir_path, rawdata_path, num_clients, niid, balance, partition, backdoor_rate, target_y, adversary_num)
    print("Done!")
