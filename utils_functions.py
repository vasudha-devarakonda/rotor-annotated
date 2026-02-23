""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy as np

import torch
torch.manual_seed(seed=42)
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.data import FlexibleDataLoader
import rotor
dataset_names = ['cifar10', 'cifar100', 'imagenet', 'kmnist', 'country211', 'flowers102', 'caltech256']

def get_network(args, num_classes=100, use_checkpoint=False, device=None):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_classes)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_classes)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_classes)
    elif args.net == 'googlenet':
        net = rotor.models.googlenet()
    elif args.net == 'inceptionv3':
        net = rotor.models.inception_v3()
    elif args.net == 'inceptionv4':
        net = rotor.models.inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_classes)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_classes)
    elif args.net == 'resnetcustom':
        from models.resnet import resnetcustom
        net = resnetcustom(num_classes)
    elif args.net == 'resnet18':
        net = rotor.models.resnet18()
    elif args.net == 'resnet34':
        net = rotor.models.resnet34()
    elif args.net == 'resnet50':
        net = rotor.models.resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152':
        net = rotor.models.resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_classes)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_classes)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_classes)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_classes)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_classes)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_classes)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_classes)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_classes)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_classes)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_classes)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_classes)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(num_classes)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_classes)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_classes)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(num_classes)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(num_classes)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_classes)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_classes)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_classes)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_classes)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_classes)
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(num_classes)
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18(num_classes)
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34(num_classes)
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50(num_classes)
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101(num_classes)
    elif args.net == 'lenet':
        from models.lenet import lenet
        net = lenet(num_classes)
    elif args.net.startswith("GPT"):
        from models.gpt import get_GPT
        net = get_GPT(model=args.net)
    elif args.net == 'alexnet':
        from models.alexnet import alexnet
        net = alexnet(num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        # net = net.cuda()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)

    return net

class GrayToRGB:
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

def get_training_dataloader(mean, std, batch_size=16, num_workers=0, shuffle=False, name="cifar100", dynamic_batch_size=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train_cifar100 = transforms.Compose([
        #transforms.ToPILImage(),
        GrayToRGB(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        GrayToRGB(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    assert(name in dataset_names)
    if name == 'cifar100':
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar100)
        num_classes = len(cifar100_training.classes)
        # if dynamic_batch_size:
        #     cifar100_training_loader = FlexibleDataLoader(
        #         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     return cifar100_training_loader, num_classes
        
        cifar100_training_loader = DataLoader(
            cifar100_training, shuffle=shuffle, num_workers=0, batch_size=batch_size, drop_last=True )
        return cifar100_training_loader, num_classes
    
    if name == 'imagenet':
        imagenet_training = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform_train)
        num_classes = len(imagenet_training.classes)
        # if dynamic_batch_size:
        #     imagenet_training_loader = FlexibleDataLoader(
        #         imagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     return imagenet_training_loader, num_classes

        imagenet_training_loader = DataLoader(
            imagenet_training, shuffle=shuffle, num_worker=0, batch_size=batch_size)
        return imagenet_training_loader, num_classes
    
    if name == 'kmnist':
        kmnist_training = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform_train)
        num_classes = len(kmnist_training.classes)
        # if dynamic_batch_size:
        #     kmnist_training_loader = FlexibleDataLoader(
        #         kmnist_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     return kmnist_training_loader, num_classes
            
        kmnist_training_loader = DataLoader(
            kmnist_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return kmnist_training_loader, num_classes

    # if name == 'country211':
    #     country211_training = torchvision.datasets.Country211(root='./data', split='train', download=True, transform=transform_train)
    #     num_classes = len(country211_training.classes)
    #     country211_training_loader = DataLoader(
    #         country211_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #     return country211_training_loader, num_classes
    
    if name == "flowers102":
        flowers102_training = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform_train)
        # if dynamic_batch_size:
        #     flowers102_training_loader = FlexibleDataLoader(
        #         flowers102_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     num_classes = len(set(label for _, label in flowers102_training_loader.dataset))
        #     return flowers102_training_loader, num_classes
            
        flowers102_training_loader = DataLoader(
            flowers102_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        num_classes = len(set(label for _, label in flowers102_training_loader.dataset))
        # print("num_classes: ", num_classes)
        # print(set(label for _, label in flowers102_training_loader.dataset))
        # print("target min: ", min(label for _, label in flowers102_training_loader.dataset))
        # print("target max: ", max(label for _, label in flowers102_training_loader.dataset))
        return flowers102_training_loader, num_classes
    
    if name == "caltech256":
        def collate_fn(batch):
            for x in batch:
                print(x)
            return {
                'pixel_values': torch.stack([x[0] for x in batch]),
                'labels': torch.tensor([x[1] for x in batch])
            }
        caltech256_training = torchvision.datasets.Caltech256(root='./data', download=True, transform=transform_train)
        # if dynamic_batch_size:
        #     caltech256_training_loader = FlexibleDataLoader(
        #         caltech256_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     num_classes = 257
        #     return caltech256_training_loader, num_classes
        num_workers=0
        caltech256_training_loader = DataLoader(
            caltech256_training, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        num_classes = 257
        return caltech256_training_loader, num_classes
    
    if name == "cifar10":
        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        num_classes = len(cifar10_training.classes)
        # if dynamic_batch_size:
        #     cifar10_training_loader = FlexibleDataLoader(
        #         cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        #     return cifar10_training_loader, num_classes
        cifar10_training_loader = DataLoader(
            cifar10_training, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        return cifar10_training_loader, num_classes
    
    assert(False)

def get_test_dataloader(mean, std, batch_size=16, num_workers=0, shuffle=True, name="cifar100"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test_cifar100 = transforms.Compose([
        GrayToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        #transforms.ToPILImage(),
        GrayToRGB(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    assert(name in dataset_names)
    if name == 'cifar100':
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar100)
        num_classes = len(cifar100_test.classes)
        cifar100_test_loader = DataLoader(
            cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)

        return cifar100_test_loader, num_classes
    
    if name == 'imagenet':
        imagenet_test = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform_test)
        num_classes = len(imagenet_test.classes)
        imagenet_test_loader = DataLoader(
            imagenet_test, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        return imagenet_test_loader, num_classes
    
    if name == 'kmnist':
        kmnist_test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = len(kmnist_test.classes)
        kmnist_test_loader = DataLoader(
            kmnist_test, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        return kmnist_test_loader, num_classes
    
    # if name == 'country211':
    #     country211_test = torchvision.datasets.Country211(root='./data', split='test', download=True, transform=transform_test)
    #     num_classes = len(country211_test.classes)
    #     country211_test_loader = DataLoader(
    #         country211_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #     return country211_test_loader, num_classes
    
    if name == "flowers102":
        flowers102_training = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform_test)
        flowers102_training_loader = DataLoader(
            flowers102_training, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        num_classes = len(set(label for _, label in flowers102_training_loader.dataset))
        return flowers102_training_loader, num_classes
    
    if name == "caltech256":
        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x[0] for x in batch]),
                'labels': torch.tensor([x[1] for x in batch])
            }
        caltech256_training = torchvision.datasets.Caltech256(root='./data', download=True, transform=transform_test)
        caltech256_training_loader = DataLoader(
            caltech256_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        num_classes = 257
        return caltech256_training_loader, num_classes
    
    if name == "cifar10":
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = len(cifar10_test.classes)
        cifar10_test_loader = DataLoader(
            cifar10_test, shuffle=shuffle, num_workers=0, batch_size=batch_size)
        return cifar10_test_loader, num_classes
    
    assert(False)
    return None, None

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]