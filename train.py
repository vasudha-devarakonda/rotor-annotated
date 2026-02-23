# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import csv
import torch
torch.manual_seed(seed=42)
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from torchsummary import summary
from tqdm import tqdm
import rotor
from torch.utils.data import DataLoader


from conf import settings
import subprocess

from utils_functions import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


torch.backends.cuda.matmul.allow_tf32 = False  # Disable TensorFloat-32 optimizations
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Disable FP16 optimizations

# def get_model_size_bytes(model):
#     """Calculate model parameter size in bytes"""
#     total_params = 0
#     for param in model.parameters():
#         total_params += param.numel()
#     # Assuming float32 parameters (4 bytes each)
#     return total_params * 4


import os
import csv
import re

import re

def process_recomp_file(stats_file):
    num_recomp = 0
    recomp_list = []
    recomp_size = 0.0
    total_recomp_ops = 0
    if "resnet" in args.net.lower():
        class_map = {
            "Bottleneck": 10,
            "Conv2d": 1,
            "ReLUatEnd": 2,
            "BasicBlock": 7,
            "conv3x3": 1,
            "conv1x1": 1
        }
    elif "googlenet" in args.net.lower():
        class_map = {
            "Inception": 23,
            "BasicConv2d": 3,
            "MaxPool2d": 1,
            "AdaptiveAvgPool2d": 7,
            "Dropout2d": 1, 
            "Linear": 1, 
            "Flatten": 1, 
            
        }
    elif "inceptionv3" in args.net.lower():
        class_map = {
            "InceptionA": 23,
            "BasicConv2d": 3,
            "ReLUatEnd": 2,
            "InceptionB": 14,
            "InceptionC": 32, 
            "InceptionD": 20, 
            "InceptionE": 31, 
            "InceptionAux": 8
        }
    elif "inceptionv4" in args.net.lower(): 
        class_map = {
            "Inception_Stem": 38,
            "BasicConv2d": 3,
            "InceptionA": 23,
            "ReductionA": 14,
            "InceptionB": 32, 
            "ReductionB": 20, 
            "InceptionC": 34
        }
    elif "gpt2" in args.net.lower():
        class_map = {
            "GPT2Block": 1
        }
    else:
        class_map = {}
    with open(stats_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if line.startswith("num_recomp"):
            num_recomp = int(line.split(":")[1].strip())

        elif line.startswith("recomp:"):
            recomp_line = line.split(":", 1)[1].strip()
            recomp_list = re.findall(r"<class '([\w\.]+)'>", recomp_line)

        elif line.startswith("recomp_size"):
            recomp_size = float(re.findall(r"([\d\.]+)", line)[0])

    for cls in recomp_list:
        for key, val in class_map.items():
            if key in cls:
                total_recomp_ops += val
                break

    return num_recomp, total_recomp_ops, recomp_size


def should_increase_batch_size(b_incr: float):
    if b_incr <= 1:
        return False
    
    current_gpu_memory = get_memory_usage(args.gpu_device)  # Get current GPU memory usage in MB
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert to MB

    if current_gpu_memory * b_incr < total_gpu_memory:
        return True
    
    print(f"Should not increase batch size to avoid OOM. Current GPU memory usage: {current_gpu_memory:.2f} MB, Total GPU memory: {total_gpu_memory:.2f} MB")
    return False

    
def get_layer_output_sizes(model, input_tensor):
    """
    Calculates the byte size of the output of each layer in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): A sample input tensor.

    Returns:
        list: A list of output sizes in bytes for each layer.
    """

    output_sizes = []
    hooks = []

    def hook_fn(module, input, output):
        output_sizes.append(output.numel() * output.element_size())

    # Register hooks for each layer
    for module in model.modules():
        if not isinstance(module, torch.nn.Sequential):  # Avoid double-counting for Sequential layers
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass to get output sizes
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return output_sizes

# def get_memory_usage(gpu:int):
#     """
#     Returns the current memory usage of the GPU.
#     """
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
#     else:
#         return 0

def get_memory_usage(gpu: int):
    """
    Returns the current memory usage of the GPU using nvidia-smi.
    """
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", "-i", str(gpu)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            memory_usage = int(result.stdout.strip())  # Memory usage in MB
            return memory_usage
        except Exception as e:
            print(f"Error while fetching GPU memory usage: {e}")
            return 0
    else:
        return 0

def train(epoch, training_loader):
    avg_mem_forward  = 0
    avg_mem_backward = 0  
    avg_time =0 
    start = time.time()
    net.train()
    peak_mem = 0
    print(len(training_loader))
    print(args.iter_limit)
    
    num_iters = min(len(training_loader), args.iter_limit) if args.iter_limit != -1 else len(training_loader)
    num_recomp, recomp_size = 0, 0
    progress_bar = tqdm(total=num_iters, desc=f"Training Epoch {epoch}", leave=False)
    
    for batch_index, (images, labels) in enumerate(training_loader):
        torch.cuda.reset_peak_memory_stats()
        if args.gpu:
            # labels = labels.cuda()
            # images = images.cuda()
            labels = labels.to(device)
            images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        avg_mem_forward += torch.cuda.memory_allocated()
        # torch.cuda.empty_cache()
        # if (args.print_memory): 
        #     print(f"Allocated GPU memory after forward: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        #     print(f"Max Allocated GPU memory after forward: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
            # clear stats
            # torch.cuda.memory.reset_accumulated_memory_stats()
        # torch.cuda.empty_cache()
        loss = loss_function(outputs, labels)
        
        if (args.empty_cache):
            torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        avg_mem_backward += torch.cuda.memory_allocated()
        if (args.print_memory and batch_index == 2):
            print(f"Allocated GPU memory after backward: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            print(f"Max Allocated GPU memory after backward: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
            # clear stats
            torch.cuda.memory.reset_accumulated_memory_stats()
        # torch.cuda.empty_cache()
        
        if (not args.no_iter_progress):
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(1)

        if epoch <= args.warm:
            warmup_scheduler.step()
            
        if args.iter_limit != -1 and batch_index >= args.iter_limit - 1:
            break
        peak_mem += torch.cuda.max_memory_allocated()


    finish = time.time()
    # avg_mem_forward = avg_mem_forward / num_iters
    # avg_mem_backward = avg_mem_backward /num_iters
    peak_mem = peak_mem / num_iters
    time_taken = finish - start
    #avg mwm fwd (GB), avg mem bwd (GB) and avg peak mem (GB), time_taken(seconds)
    # data.append((avg_mem_forward/(1024**3), avg_mem_backward/(1024**3), peak_mem/(1024**3), time_taken))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, time_taken))
    progress_bar.close()


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            # images = images.cuda()
            # labels = labels.cuda()
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    
    loss = test_loss / len(cifar100_test_loader.dataset)
    accuracy = correct.float() / len(cifar100_test_loader.dataset)
    time_consumed = finish - start
    
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss,
        accuracy,
        time_consumed
    ))
    print()

    #add informations to tensorboard
    # if tb:
    # writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return loss, accuracy, time_consumed

def model_converges(measurements: list, threshold: float = 0.01) -> bool:
    max_measurement = max(measurements)
    min_measurement = min(measurements)
    if max_measurement - min_measurement < threshold:
        return True
    return False

def get_model_and_a_sample(args, num_classes, test_loader):
    net = get_network(args, num_classes, args.c, device=device)
    setattr(net, "model_name", args.net)
    sample = next(iter(test_loader))
    # sample = sample[0].cuda()
    sample = sample[0].to(device)
    return net, sample

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    # parser.add_argument('-gpu-device', type=int, default=0, help='device id to use')
    parser.add_argument('-gpu-device', type=str, default="0", help='device id to use')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-b-incr', type=float, default=1, help='batch size increment factor')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-lr-decay', type=float, default=0.5, help='learning rate decay, set to 1 to not decay')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-e', type=int, default=2, help='Number of training epochs')
    parser.add_argument('-d', type=str, default='cifar100', help='dataset')
    parser.add_argument('-i', type=int, default=0, help='experiment iteration')
    parser.add_argument('-hbe', type=int, default=-1, help='half batch size at epoch e')
    parser.add_argument('-elim', type=int, default=-1, help='early stopping at epoch e in case -e is -1')
    parser.add_argument('-write-stats', type=bool, default=False, help='write stats to file')
    parser.add_argument('-m-interval', type=int, default=20, help='milestone interval, -1 to use settings.MILESTONES')
    parser.add_argument('-iter-limit', type=int, default=-1, help='limit the number of iterations per epoch, -1 to use full dataset')
    parser.add_argument('-c', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-empty-cache', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-no-eval', action='store_true', default=False, help='do evaluation after training')
    
    parser.add_argument('-rockmate', action='store_true', default=False, help='rockmate optimization')
    parser.add_argument('-hiremate', action='store_true', default=False, help='hiremate optimization')
    parser.add_argument('-offmate', action='store_true', default=False, help='offmate optimization')
    parser.add_argument('-checkmate', action='store_true', default=False, help='checkmate optimization')
    parser.add_argument('-segment-ilp', action='store_true', default=False, help='segment ilp optimization')
    parser.add_argument('-no-recompute', action='store_true', default=False, help='disable recomputation in rockmate')
    
    parser.add_argument('-budget', type=float, default=3, help='memory budget')
    parser.add_argument('-filename', type=str,default="results", help='file_to_write')
    parser.add_argument('-ilps', type=int, default=2, help='num,ber of ilps')
    parser.add_argument('-print-memory', action='store_true', default=False, help='print memory usage after each epoch')
    parser.add_argument('-no-iter-progress', action='store_true', default=False, help='show progress bar for iterations')
    parser.add_argument('-fragmentation-hook', action='store_true', default=False, help='use fragmentation hook in rockmate')
    args = parser.parse_args()

    # Can only use one memory mode
    assert args.rockmate + args.checkmate + args.segment_ilp + args.no_recompute + args.hiremate + args.offmate <= 1, "Can only use one memory optimization method at a time"

    # os.environ['PYTORCH_MODEL_NAME'] = f"{args.net}_{args.b}"
    # torch.initialize(f"{args.net}_{args.b}")
    data = []
    stats_file_name = f'stats_{args.net}_{args.d}_b{args.b}_b-incr{args.b_incr}_e{args.e}_lr{args.lr}_lr-decay{args.lr_decay}_m-interval{args.m_interval}_i{args.i}.txt'
    stats_file_path = os.path.join("stats", stats_file_name)
    batch_size = args.b
    
    # set GPU device
    # data.append(args.net)
    # data.append(args.budget)
    offline_time = 0
    device = torch.device("cpu")
    use_gpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda:'+args.gpu_device)
        # torch.cuda.set_device(args.gpu_device)
        torch.cuda.set_device(device)
        print(f"Using GPU device {torch.cuda.current_device()}")
        device = torch.device("cuda")
        use_gpu = True
    else:
        print("No GPU available, using CPU")
    
    # with open(stats_file_path, 'w') as f:
    #     f.write("epoch,iteration,batch_size,loss\n")
    print(f"batch size is {args.b}")
    if args.write_stats:
        if not os.path.exists("stats"):
            os.makedirs("stats")
        with open(stats_file_path, 'w') as f:
            f.write(f"epoch {args.m_interval},loss {args.m_interval},accuracy {args.m_interval},time_consumed(s) {args.m_interval},batch_size {args.m_interval},memory_usage(MB) {args.m_interval}\n")
    
    # get dataset config

    #data preprocessing:
    cifar100_training_loader, num_classes_train = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=True,
        name=args.d
    )
    
    cifar100_test_loader, num_classes_test = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=True,
        name=args.d
    )
    
    assert(num_classes_train == num_classes_test)
    mega = 1024 **2
    time_start = time.time()
    net, sample = get_model_and_a_sample(args, num_classes_train, cifar100_test_loader)
    print(net)
    net = rotor.Checkpointable(net)
    net.measure(sample)
    net.compute_sequence(mem_limit = args.budget * mega)
    offline_time = time.time() - time_start
    print(f"Offline model construction time: {offline_time:.2f} seconds")
    if net is None: 
        print("model not found")
        exit(1)
    params = list(net.parameters())
    # data.append(args.ilps)
    if not params:
        print("no paramters")
        with open(args.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        sys.exit(1)
    # data.append(offline_time)
    print(f"Model parameters use {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB of GPU memory")

    data_loader_iter = iter(cifar100_training_loader)
    # input_shape = next(data_loader_iter)[0].shape  # Get the
    input_batch = next(data_loader_iter)[0]
    if args.gpu:
        # input_batch = input_batch.cuda()
        input_batch = input_batch.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.lr_decay) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    assert(args.m_interval >= -1)
    # for epoch in range(1, settings.EPOCH + 1):
    epoch = 1
    mistep = 0
    # prev_acc = 0.0
    accuracies = list()
    min_epochs = 10
    
    current_mile_stone = settings.MILESTONES[0]
    if args.m_interval != -1:
        current_mile_stone = 0 + args.m_interval

    current_batch_size = args.b
    
    training_loader, num_classes_train = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=0,
        batch_size=current_batch_size,
        shuffle=True,
        name=args.d
    )
    num_recomp, size_recomp = 0, 0
    print("Start training {} with batch size {}".format(args.net, current_batch_size))
    time_start_train = time.time()
    while True:
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if epoch == current_mile_stone:
            if args.m_interval == -1:
                current_mile_stone = settings.MILESTONES[settings.MILESTONES.index(current_mile_stone) + 1]
            else:
                assert(args.m_interval >= 0)
                current_mile_stone += args.m_interval
            
            if should_increase_batch_size(args.b_incr):
                current_batch_size = int(current_batch_size * args.b_incr)

            print(f"Batch size increased to {current_batch_size}")
            training_loader, num_classes_train = get_training_dataloader(
                settings.CIFAR100_TRAIN_MEAN,
                settings.CIFAR100_TRAIN_STD,
                num_workers=0,
                batch_size=current_batch_size,
                shuffle=True,
                name=args.d
            )
        
        start_time = datetime.now()
        train(epoch, training_loader)
        time_consumed = (datetime.now() - start_time).total_seconds()
        print(f"Time consumed for training epoch {epoch}: {time_consumed:.2f}s")
        
        if not args.no_eval:
            eval_start_time = datetime.now()
            loss, acc, eval_time_consumed = eval_training(epoch)
            eval_time_consumed = (datetime.now() - eval_start_time).total_seconds()
            memory_usage = get_memory_usage(args.gpu_device)
            if args.write_stats:
                with open(stats_file_path, 'a') as f:
                    f.write('{},{:.4f},{:.4f},{:.2f},{},{}\n'.format(
            epoch,
            loss,
            acc,
            time_consumed,
            current_batch_size,
            memory_usage,
            ))
            print(f"Time consumed for evaluating epoch {epoch}: {eval_time_consumed:.2f}s")
            accuracies.append(acc)
            if len(accuracies) > min_epochs:
                accuracies.pop(0)
        
        epoch += 1
        
        # early stopping
        if (args.elim != -1 and args.e == -1) and epoch > args.elim:
            print('Early stopping at epoch {}'.format(epoch))
            break
        if args.e == -1 and not args.no_eval:
            if len(accuracies) >= min_epochs and model_converges(accuracies):
                print('Model converged at epoch {}'.format(epoch))
                break
            continue
        assert(args.e > 0)
        # prev_acc = acc
        if epoch > args.e:
            break
    time_train = time.time() - time_start_train
    stats_file = "test.txt"
    num_recomp, total_recomp_ops, recomp_size = process_recomp_file(stats_file)
    csv_file = "rotor_values.csv"
    header = [
        "model",
        "batch",
        "num_recomp",
        "total_recomp_ops",
        "recomp_size_MB",
        "max_allocated_gpu_MB",
        "offline_time_s",
        "training_time_s",
        "budget"
    ]
    row = [
        args.net,
        args.b,
        num_recomp,
        total_recomp_ops,
        recomp_size,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        offline_time,
        time_train,
        args.budget
    ]
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)