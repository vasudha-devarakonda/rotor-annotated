# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime
import csv
import torch
torch.manual_seed(seed=42)
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from torchsummary import summary
from tqdm import tqdm

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader


from conf import settings
import subprocess

from utils_functions import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
import rotor 
torch.backends.cuda.matmul.allow_tf32 = False  # Disable TensorFloat-32 optimizations
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Disable FP16 optimizations


from rockmate import Rockmate

# def get_model_size_bytes(model):
#     """Calculate model parameter size in bytes"""
#     total_params = 0
#     for param in model.parameters():
#         total_params += param.numel()
#     # Assuming float32 parameters (4 bytes each)
#     return total_params * 4

class CheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module, use_reentrant: bool) -> None:
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        # Closure allows forwarding kwargs through checkpoint safely.
        def run_fn(*inputs):
            return self.module(*inputs, **kwargs)

        return checkpoint(run_fn, *args, use_reentrant=self.use_reentrant)
    
def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)

def apply_checkpoint_wrappers(model: nn.Module, names: Sequence[str], use_reentrant: bool) -> None:
    for name in sorted(names, key=lambda n: n.count("."), reverse=True):
        original = model.get_submodule(name)
        wrapped = CheckpointWrapper(original, use_reentrant=use_reentrant)
        replace_module(model, name, wrapped)

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
    
def get_optimizer_state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total

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
        # optimizer.zero_grad()
        avg_mem_backward += torch.cuda.memory_allocated()
        if (batch_index < 5):
            print("----------")
            print(f"Input size (in MB): {images.numel() * images.element_size() / 1024 / 1024:.2f} MB")
            print(f"Labels size (in MB): {labels.numel() * labels.element_size() / 1024 / 1024:.2f} MB")
            print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")
            print(f"Parameter size (in MB): {sum(p.numel() * p.element_size() for p in net.parameters()) / 1024 / 1024:.2f} MB")
            print(f"Allocated GPU memory after backward: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            print(f"Max Allocated GPU memory after backward: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
            # clear stats
            torch.cuda.memory.reset_accumulated_memory_stats()
            
            # report memory usage of optimizer state
            print(f"Optimizer state memory usage {get_optimizer_state_bytes(optimizer) / 1024 / 1024:.2f} MB")
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
    model_wrapper = None
    
    if args.rockmate:
        model_wrapper = getRockmateModel
    elif args.checkmate:
        model_wrapper = getModelFromConfig
    elif args.segment_ilp:
        model_wrapper = getSegmentILPModel
    elif args.greedy_search:
        model_wrapper = getGreedySearchModel
    elif args.recompute_op_list:
        model_wrapper = getRecompOpListModel
    elif args.no_recompute:
        model_wrapper = getNoRecomputeModel
    elif args.offload_all:
        model_wrapper = getOffloadAllModel
    elif args.hiremate:
        model_wrapper = getHiremateModel
    elif args.offmate:
        model_wrapper = getOffmateModel
    else:
        print("No scheduler applied")


    if args.offmate:
        net = get_network(args, num_classes, args.c, device=torch.device("cpu"))
        setattr(net, "model_name", args.net)
        sample = next(iter(test_loader))
        sample = sample[0]
        if model_wrapper is not None:
            assert type(args.budget) == float or type(args.budget) == int, "Please provide a memory budget in bytes"
            net = model_wrapper(net, sample, args.budget)
        # net = net.cuda()
        # sample = sample.cuda()
        net = net.to(device)
        sample = sample.to(device)
        return net, sample

    net = get_network(args, num_classes, args.c, device=device)
    setattr(net, "model_name", args.net)
    sample = next(iter(test_loader))
    # sample = sample[0].cuda()
    sample = sample[0].to(device)
    if model_wrapper is not None:
        # if args.no_recompute:
        #     net = model_wrapper(net, sample)
        # else:
        net = model_wrapper(net, sample, args.budget)
        
    if args.module_checkpoint:
        # load list of checkpointed modules from json file
        with open(args.module_checkpoint_path, 'r') as f:
            checkpointed_modules = json.load(f)
        apply_checkpoint_wrappers(net, checkpointed_modules, use_reentrant=True)
        
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
    parser.add_argument('-offload-all', action='store_true', default=False, help='offload all activations to CPU')
    parser.add_argument('-module-checkpoint', action='store_true', default=False, help='use module checkpointing in rockmate from search results, user profvide path to .json file')
    parser.add_argument('-module-checkpoint-path', type=str, default="", help='path to the .json file with module checkpointing results')
    parser.add_argument('-greedy-search', action='store_true', default=False, help='get greedy search model without running the search, user must provide path to trace files with environment variables CUSTOM_TRACE_FILE_PATH and MODULE_OP_TRACE_FILE_PATH')
    parser.add_argument('-recompute-op-list', action='store_true', default=False, help='get model with recompute op list without running the search, user must provide path to the recompute op list file with environment variable RECOMPUTE_OP_LIST_FILE_PATH')
    
    parser.add_argument('-budget', type=float, default=3, help='memory budget')
    parser.add_argument('-filename', type=str,default="results", help='file_to_write')
    parser.add_argument('-ilps', type=int, default=2, help='num,ber of ilps')
    parser.add_argument('-print-memory', action='store_true', default=False, help='print memory usage after each epoch')
    parser.add_argument('-no-iter-progress', action='store_true', default=False, help='show progress bar for iterations')
    parser.add_argument('-fragmentation-hook', action='store_true', default=False, help='use fragmentation hook in rockmate')
    args = parser.parse_args()

    # Can only use one memory mode
    assert args.rockmate + args.checkmate + args.segment_ilp + args.no_recompute + args.hiremate + args.offmate + args.offload_all + args.module_checkpoint + args.greedy_search + args.recompute_op_list <= 1, "Can only use one memory optimization method at a time"
    
    if args.module_checkpoint:
        assert args.module_checkpoint_path, "Please provide path to module checkpointing results with -module-checkpoint-path"
        assert os.path.exists(args.module_checkpoint_path), f"Module checkpointing results file not found at {args.module_checkpoint_path}"

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
    # if args.rockmate: 
    #     model = get_network(args, num_classes_train, args.c)
    #     sample = next(iter(cifar100_test_loader))
    #     sample = sample[0].cuda()
    #     time_start = time.time()
    #     net = getRockmateModel(model,sample, args.budget)
    #     offline_time = time.time() - time_start
    #     # from rockmate import PureCheckmate
    #     # net_checkpoint = PureCheckmate(model, sample, args.budget*(1024**3)) 
    #     # del net_checkpoint
    #     del sample 
    #     del model 
    # elif args.checkmate: 
    #     model = get_network(args, num_classes_train, args.c)
    #     sample = next(iter(cifar100_test_loader))
    #     sample = sample[0].cuda()
    #     time_start = time.time()
    #     net = getModelFromConfig(model, sample, args.budget) 
    #     offline_time = time.time() - time_start
    #     del sample 
    #     del model 
    # elif args.segment_ilp:
    #     model = get_network(args, num_classes_train, args.c)
    #     sample = next(iter(cifar100_test_loader))
    #     sample = sample[0].cuda()
    #     time_start = time.time()
    #     net = getSegmentILPModel(model, sample, args.budget) 
    #     offline_time = time.time() - time_start
    #     del sample 
    #     del model
    # elif args.no_recompute:
    #     model = get_network(args, num_classes_train, args.c)
    #     sample = next(iter(cifar100_test_loader))
    #     sample = sample[0].cuda()
    #     time_start = time.time()
    #     net = getNoRecomputeModel(model, sample, args.budget) 
    #     offline_time = time.time() - time_start
    #     del sample 
    #     del model
    # else: 
    #     net = get_network(args, num_classes_train, args.c)

    net, sample = get_model_and_a_sample(args, num_classes_train, cifar100_test_loader)
    net = rotor.Checkpointable(net)
    net.measure(sample)
    mega = 1024**2
    time_start = time.time()
    net.compute_sequence(mem_limit = args.budget * mega)
    offline_time = time.time() - time_start
    schedule_str = str(net.sequence)

    # Verwijder de vierkante haken
    schedule_str = schedule_str.strip("[]")

    # Split op komma's en strip whitespace
    schedule = [x.strip() for x in schedule_str.split(",")]

    print(schedule)
   
    l_index = schedule.index("L")
    
    after_L = schedule[l_index + 1:]
    f_after_L = [x for x in after_L if x.startswith("F")]

    indices = [int(x.split("_")[1]) for x in f_after_L]
    print(indices)
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
    print("Start training {} with batch size {}".format(args.net, current_batch_size))
    time_start_train = time.time()
    while True:
    # for epoch in range(1, args.e + 1):
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
        # try:
        #     torch.cuda.memory._record_memory_history(max_entries=1000000)
        # except Exception as e:
        #     print(f"Could not record memory history (Requires PyTorch 2.1+)xxx: {e}")
        train(epoch, training_loader)
        # torch.cuda.memory._dump_snapshot(f"./memory_snapshots/memory_snapshot_{args.net}_{current_batch_size}.pickle")
        time_consumed = (datetime.now() - start_time).total_seconds()
        print(f"Time consumed for training epoch {epoch}: {time_consumed:.2f}s")
        
        if not args.no_eval:
            eval_start_time = datetime.now()
            loss, acc, eval_time_consumed = eval_training(epoch)
            eval_time_consumed = (datetime.now() - eval_start_time).total_seconds()
            memory_usage = get_memory_usage(args.gpu_device)
            
            # save accuracy of this epoch
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
        
        # Model converges
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
    # data.append(time_train)
    # with open(args.filename, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(data)

    # Print max allocated memory

    print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"Training completed in {time_train:.2f} seconds")
    print(f"Offline time: {offline_time:.2f} seconds")


# export CUSTOM_TRACE_FILE_PATH=$HOME/pytorch-cifar100/tmp_trace/operator_trace_resnet18_256_RTX6000.txt
# python train.py -gpu -net resnet18 -b 256 -e 2 -iter-limit 1 -no-eval -segment-ilp -budget 0.7  -gpu-device 0
# python train.py -gpu -net resnet18 -b 256 -e 2 -iter-limit 1 -no-eval -segment-ilp -budget 0.7  -gpu-device 0