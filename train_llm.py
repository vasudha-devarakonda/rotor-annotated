import torch
torch.manual_seed(seed=42)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    # TrainerCallback
)
from datasets import load_dataset
import os
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import time
import rotor

import csv

'''
Some small llm
microsoft/phi-2
openlm-research/open_llama_3b
EleutherAI/gpt-neo-125M
facebook/opt-350m
EleutherAI/pythia-160m
'''

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        out = self.model(**inputs)
        return out.logits   # remove HF dict → required for stable dynamo export



# class EmptyCacheCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model_size_bytes(model):
    """Calculate model parameter size in bytes"""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    # Assuming float32 parameters (4 bytes each)
    return total_params * 4

class CustomDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, device='cpu'):
        self.device = device
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self) -> int:
        return len(self.encodings.input_ids)



def train_loop(model, train_dataset, batch_size, epochs):
    # write my own training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    num_iters = min(len(dataloader), args.iter_limit) if args.iter_limit != -1 else len(dataloader)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start = time.time()
        progress_bar = tqdm(total=num_iters, desc=f"Training Epoch {epoch}", leave=False)
        for step, batch in enumerate(dataloader):
            if step >= num_iters - 1:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            tensor = batch['input_ids']
            labels = tensor
            
            outputs = model(tensor)

            logits = outputs
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            # loss.backward()
            # outputs = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(1)
            
            if step % 100 == 0:
                time_taken = time.time() - start
                print(f"Step {step} Time elapsed: {time_taken:.2f}s")
        finish = time.time()
        time_taken = finish - start
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, time_taken))
        progress_bar.close()

dataset_name_map = {
    "databricks-dolly-15k": "databricks/databricks-dolly-15k",
    "databricks-databricks-dolly-15k": "databricks/databricks-dolly-15k",
    "dolly-15k": "databricks/databricks-dolly-15k",
}
def prepare_data(dataset_name) -> List[str]:
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name_map[dataset_name], split="train")

    dataset = dataset.train_test_split(test_size=0.1)  # Adjust test_size as needed

    # Extract instruction-response pairs for each split
    train_texts = [
        f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
        for item in dataset['train']
    ]
    eval_texts = [
        f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
        for item in dataset['test']
    ]

    return train_texts, eval_texts

def get_model_and_a_sample(args, test_loader, config, model_name):
    if model_name == "openai-community/gpt2": 
        from rotor.models.gpt2 import gpt2
        net = gpt2(config)
    if model_name == "EleutherAI/pythia-160m":
        from rotor.models.pythia import pythia
        net = pythia(config)
    if model_name == "facebook/opt-350m":
        from rotor.models.opt import opt
        net = opt(config)
    net = net.to(device)    
    return net


def test_model_memory(model, input_ids, labels, device):
    """
    Runs forward + backward pass on a batch, computes loss, and returns max GPU memory used.
    
    Args:
        model: HuggingFace or sequential OPT model
        input_ids: tensor [batch_size, seq_len]
        labels: tensor [batch_size, seq_len]
        device: torch device (cpu or cuda)
    
    Returns:
        loss value, max GPU memory in MB
    """
    torch.cuda.reset_peak_memory_stats(device)
    model.train()  # enable gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


    optimizer.zero_grad()
    outputs = model(input_ids)

    # Extract logits for HF model or use direct tensor for sequential wrapper
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    torch.cuda.reset_peak_memory_stats(device)
    stats = torch.cuda.memory_stats(device)
    mem_before = stats["allocated_bytes.all.current"]
    m_b = torch.cuda.memory_allocated()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    m_a = torch.cuda.memory_allocated()
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    mem_after = stats["allocated_bytes.all.current"]

    loss.backward()
    optimizer.step()

    # Max GPU memory
    print(f"{(m_a-m_b)/(1024**2)} we have")
    return loss.item(), (mem_after - mem_before)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    # parser.add_argument('-gpu-device', type=int, default=0, help='device id to use')
    parser.add_argument('-gpu-device', type=str, default="0", help='device id to use')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-e', type=int, default=2, help='number of epochs')
    parser.add_argument('-d', type=str, default='databricks-dolly-15k', help='dataset')
    parser.add_argument('-iter-limit', type=int, default=-1, help='limit the number of iterations per epoch, -1 to use full dataset')
    parser.add_argument('-c', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-empty-cache', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-no-eval', action='store_true', default=False, help='do evaluation after training')
    parser.add_argument('-budget', type=float, default=3, help='memory budget')
    parser.add_argument('-filename', type=str,default="results", help='file_to_write')
    parser.add_argument('-ilps', type=int, default=2, help='num,ber of ilps')
    # parser.add_argument('-print-memory', action='store_true', default=False, help='print memory usage after each epoch')
    # parser.add_argument('-no-iter-progress', action='store_true', default=False, help='show progress bar for iterations')
    # parser.add_argument('-no-eval', action='store_true', default=False, help='do evaluation after training')
    args = parser.parse_args()
    
    # Can only use one memory mode

    # device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
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
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     # torch.cuda.set_device(args.gpu_device)
    #     # print(f"Using GPU device {torch.cuda.current_device()}")
    #     # device = torch.device("cuda")
    #     # use_gpu = True
    # else:
    #     print("No GPU available, using CPU")
    print(f"Using device: {device}")

    # Load model and tokenizer using Auto classes
    # model_name = "facebook/opt-350m"  # You can easily change this to any model
    model_name = args.net  # You can easily change this to any model
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = model.config
    model = model.to(device)
    
    
    print(config.vocab_size)
    # model_size = get_model_size_bytes(model)

    #torch.initialize(f"{args.net}_{args.b}",model_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare dataset
    train_texts, eval_texts = prepare_data(args.d)
    train_dataset = CustomDataset(train_texts, tokenizer, device=device)
    eval_dataset = CustomDataset(eval_texts, tokenizer, device=device)
    mega = 1024**2
    # Get one batch of training data
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False)
    # input_batch = next(iter(train_loader))
    # if args.gpu:
    #     input_batch = input_batch.cuda()
    # print("Sample batch keys:", batch.keys())
    # print("input_ids shape:", batch['input_ids'].shape)
    sample = next(iter(eval_loader))
    sample = {k: v.to(device) for k, v in sample.items()}
    sample = sample['input_ids']
    labels = sample.clone() 
    loss_hf, mem_loss = test_model_memory(model, sample, labels,device)
    print(f"the memroy consumption of loss {mem_loss} for {args.net}")
    time_start = time.time()
    model = get_model_and_a_sample(args, eval_loader, config, args.net)
    # loss_seq, mem_seq = test_model_memory(model, sample,labels,device)
    # print(f"Sequential wrapper {args.net}: loss={loss_seq:.6f}, max GPU memory={mem_seq:.2f} MB")
    # print(model)
    if "opt" in args.net: 
        mem_loss = 2*mem_loss
    print(f"{(mem_loss)/(1024**2)} we have")
    mem_limit_new = (args.budget * mega) - mem_loss
    print(f"the new budget is {mem_limit_new}")
    model = rotor.Checkpointable(model)
    model.measure(sample)
    model.compute_sequence(mem_limit = mem_limit_new)
    offline_time = time.time() - time_start
    model_name = "pythia"
    if "pythia" in args.net: 
        model_name = "pythia"
    if "gpt" in args.net: 
        model_name = "gpt"    
    if "opt" in args.net: 
        model_name = "opt"
    print(f"trace_{model_name}")    
    names_of_modules = model.names
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # print(names_of_modules)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    schedule_str = str(model.sequence)

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
    module_ops = [names_of_modules[i] for i in indices]
    print(module_ops)
    print(f"trace_{model_name}")
    trace_lines = open(f"trace_{model_name}").read().splitlines()
    

    module_stack = []
    cleaned_ops = []
    for k in module_ops:
        # Remove 'ResNet-Sequential-'
        if "resnet" in args.net:
            k_clean = k.replace("ResNet-Sequential-", "")
        elif "googlenet" in args.net:
            k_clean = k.replace("GoogleNet-Sequential-", "")
            k_clean = k_clean.replace("GoogleNet-", "")
        elif "inceptionv3" in args.net:
            k_clean = k.replace("Inception3-", "")
        elif "pythia" in args.net:
            k_clean = k.replace("GPTNeoXForCausalLM-Sequential-", "")
            k_clean = k_clean.replace("GPTNeoXForCausalLM-", "")
            k_clean = k_clean.replace("GPTNeoXLayerWrapper", "GPTNeoXLayer")
        elif "opt" in args.net:
            k_clean = k.replace("OPTForCausalLM-Sequential-", "")
            k_clean = k_clean.replace("OPTDecoderLayerWrapper", "OPTDecoderLayer")

        elif "gpt" in args.net:
            k_clean = k.replace("GPT2-Sequential-", "")
            # k_clean = k_clean.replace("OPTDecoderLayerWrapper", "OPTDecoderLayer")
        else:
            k_clean = k.replace("ResNet-Sequential-", "")
        # Split by '-', reverse, and join back
        parts = k_clean.split('-')
        parts.reverse()
        new_key = '-'.join(parts)
        
        cleaned_ops.append(new_key)
    # modules_fixed = [s[::-1].replace('-', '.', 1)[::-1] for s in cleaned_ops]
    modules_fixed = []

    for s in cleaned_ops:
        # replace last '-' with '.' if last part is integer
        parts = s.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            s_fixed = f"{parts[0]}.{parts[1]}"
        else:
            s_fixed = s

        # replace '=' with '.'
        s_fixed = s_fixed.replace('-.', '.')
        s_fixed = s_fixed.replace('=', '.')
        
        modules_fixed.append(s_fixed)
    # Function to extract operators for a given module
    def get_ops_for_module(module_name, trace_lines):
        ops = []
        inside_module = False
        if module_name == "OutputHead-OPTForCausalLM-output_head" and "opt" in args.net:
            module_name = "Linear-model.decoder.project_out"
        module_name_lower = module_name.lower()  # lowercase for comparison
        if module_name == "MaxPool2d-maxpool1" and "googlenet" in args.net:
            ops.append("max_pool2d_with_indices-9")
            return(ops)
        if module_name == "MaxPool2d-maxpool3" and "googlenet" in args.net:
            ops.append("max_pool2d_with_indices-56")
            return(ops)   
        if module_name == "MaxPool2d-maxpool4" and "googlenet" in args.net:
            ops.append("max_pool2d_with_indices-172")
            return(ops)         
        for i, line in enumerate(trace_lines):
            line_strip = line.strip()
            line_lower = line_strip.lower()  # lowercase line

            if f">>{module_name_lower}-" in line_lower:
                inside_module = True
            elif f"<<{module_name_lower}-" in line_lower:
                inside_module = False
            elif inside_module and line_strip.startswith("------>"):
                if i + 1 < len(trace_lines):
                    op_line = trace_lines[i + 1].strip()
                    ops.append(op_line)

        return ops

    # Collect operators for each module
    module_to_ops = {}

    for mod in modules_fixed:
        module_to_ops[mod] = get_ops_for_module(mod, trace_lines)

    # Print results
    print("=======================")
    for mod, ops in module_to_ops.items():
        if len(ops) == 0:
            print("=======================================================\n\n\n\n\n")
            print(mod)
            print("=======================================================\n\n\n\n\n")
        print(f"{mod}: {ops}")
    print("============================")
    filename = f"recomp-llm-new-runs/{model_name}_{int(args.budget)}_{int(mem_loss/(1024**3))}_recompops.txt"

    # Only make directories if filename includes a path
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Write each operator on a separate line, with a blank line between modules
    with open(filename, "w") as f:
        for ops in module_to_ops.values():
            print(len(ops))
            print("===========================================")
            for op in ops:
                f.write(op + "\n")
            f.write("----------------------\n")  # blank line to separate modules
    time_start = time.time()
    train_loop(model, train_dataset, args.b, args.e)
    time_train = time.time() - time_start
    stats_file = "test.txt"
    num_recomp=0
    total_recomp_ops =0
    recomp_size = 0
    csv_file = "rotor_values-llm-2.csv"
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
    
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -budget 10
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -no-recompute