import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import check, separate_data, split_data, save_file
from PIL import Image

random.seed(1)
np.random.seed(1)

def save_img(image_array, name):
    image_array = (image_array + 1) * 127.5  # 先转换到[0,255]范围
    image_array = image_array.clip(0, 255).astype(np.uint8)
    image_array = image_array[0]  # 去掉通道维度，变成(28,28)
    Image.fromarray(image_array, mode='L').save(name)


# add trigger to user0's train data 
def add_trigger28_sig(dict, backdoor_rate, target_y):
    feature = dict["x"]
    label = dict["y"]
    trigger = np.zeros([28,28], dtype=float)
    for i in range(28):
            for j in range(28):
                trigger[i, j] = 20/255 * np.sin(2 * np.pi * j * 6 / 28)

    num_img = feature.shape[0]
    # print(f"num_img : {num_img}")

    id_set = list(range(0, num_img))
    num_poison = int(num_img * backdoor_rate)

    poison_indices = random.sample(id_set, num_poison)

    print(f"feature i shape: {feature[0].shape}")
    for n, i in enumerate(poison_indices):
        if n == 0:
            save_img(feature[i], 'target0.png')
        feature[i] = feature[i] + trigger
        if n == 0:
            save_img(feature[i], 'target.png')
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

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
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
        train_data[i] = add_trigger28_sig(train_data[i], backdoor_rate, target_y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    for idx, test_dict in enumerate(test_data):
        test_dict = remove_target_test_data(test_dict)
        test_dict = add_trigger28_sig(test_dict, 1, target_y)
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
    backdoor_rate = 0.5
    adversary_num = 5
    dir_path = f"../MNIST_dir0.5_bdoor{backdoor_rate}_nclient_{num_clients}_sig_adv{adversary_num}/"
    target_y = 0
    aux_path = dir_path + "test/"
    server_clean_path = dir_path + "test/"

    generate_dataset(dir_path, num_clients, niid, balance, partition, backdoor_rate, target_y)