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
from torchvision.transforms import functional as F

random.seed(42)
np.random.seed(42)
GENERATOR_SEED = 42
BASE_DIR = Path(__file__).resolve().parent
TRIGGER_DIR = BASE_DIR.parents[1] / "backdoor" / "triggers"


def get_rawdata_root(dir_path):
    return globals().get("rawdata_path", os.path.join(dir_path, "rawdata"))

def save_img(image_array, name):
    image_array = (image_array + 1) * 127.5  # 先转换到[0,255]范围
    image_array = image_array.clip(0, 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    Image.fromarray(image_array).save(name)


# add trigger to user0's train data 
def add_trigger32_badnet(dict, backdoor_rate, target_y):
    feature = dict["x"]
    label = dict["y"]
    trigger = Image.open(TRIGGER_DIR / "badnet_patch_32.png")
    trigger = np.array(trigger).transpose(2, 0, 1)
    trigger = (trigger / 255.0) * 2 - 1
    mask = Image.open(TRIGGER_DIR / "mask_badnet_patch_32.png")
    mask = np.array(mask).transpose(2, 0, 1)
    mask = mask / 255.0
    #print(f"trigger shape:{trigger.shape}")

    num_img = feature.shape[0]
    # print(f"num_img : {num_img}")

    id_set = list(range(0, num_img))
    num_poison = int(num_img * backdoor_rate)

    poison_indices = random.sample(id_set, num_poison)

    print(f"feature i shape: {feature[0].shape}")
    for n, i in enumerate(poison_indices):
        if n == 0:
            save_img(feature[i], 'target0.png')
        feature[i] = feature[i] + mask * (trigger - feature[i])
        if n == 0:
            save_img(feature[i], 'target.png')
        label[i] = target_y

    return {'x': feature, 'y': label}
# 
# def generate_server_clean_dataset(dataset_image, dataset_label, server_clean_path, server_data_num):
#     server_datasets_indice = []
#     keep_mask = np.ones(len(dataset_label), dtype=bool)
# 
#     for class_idx in range(10):  # 遍历0-9类
#         # 获取当前类别的所有索引
#         class_indices = np.where(dataset_label == class_idx)[0]
#         # 随机选择100个索引（不重复）
#         selected_indices = np.random.choice(
#             class_indices, 
#             size = server_data_num // 10, 
#             replace=False
#         )
#         keep_mask[selected_indices] = False
#         server_datasets_indice += selected_indices.tolist()
#     
#     server_datasets_aux = {'x': dataset_image[server_datasets_indice], 'y': dataset_label[server_datasets_indice]}
# 
#     with open(server_clean_path  + 'server_clean.npz', 'wb') as f:
#         np.savez_compressed(f, data = server_datasets_aux)
# 
#     updated_images = dataset_image[keep_mask]
#     updated_labels = dataset_label[keep_mask]
# 
#     return updated_images, updated_labels 



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
    #dataset_image, dataset_label = generate_backdoor_test_dataset(dataset_image, dataset_label, aux_path, target_y) 
    #dataset_image, dataset_label = generate_server_clean_dataset(dataset_image, dataset_label, server_clean_path, server_data_num) 

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)

    for i in range(adversary_num):
        train_data[i] = add_trigger32_badnet(train_data[i], backdoor_rate, target_y) # alpha = 0.2
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    for idx, test_dict in enumerate(test_data):
        test_dict = remove_target_test_data(test_dict)
        test_dict = add_trigger32_badnet(test_dict, 1, target_y)
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
    dir_path = f"../Cifar10_dir0.5_bdoor{backdoor_rate}_nclient_{num_clients}_badnet_adv{adversary_num}/"
    rawdata_path = "../Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5/" + "rawdata/"
    target_y = 0
    aux_path = dir_path + "test/"
    server_clean_path = dir_path + "test/"

    generate_dataset(dir_path, num_clients, niid, balance, partition, backdoor_rate, target_y)

