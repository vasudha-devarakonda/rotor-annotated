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

from rockmateexample import getNoRecomputeModel, getRockmateModel, getSegmentILPModel, getCheckmateModel, getHiremateModel, getOffmateModel
from traincheck import getModelFromConfig
from tqdm import tqdm
import time
import rotor



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
    print(config)
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
    

    parser.add_argument('-rockmate', action='store_true', default=False, help='rockmate optimization')
    parser.add_argument('-hiremate', action='store_true', default=False, help='hiremate optimization')
    parser.add_argument('-offmate', action='store_true', default=False, help='offmate optimization')
    parser.add_argument('-checkmate', action='store_true', default=False, help='checkmate optimization')
    parser.add_argument('-segment-ilp', action='store_true', default=False, help='segment ilp optimization')
    parser.add_argument('-no-recompute', action='store_true', default=False, help='disable recomputation in rockmate')
    
    
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
    

    # model_size = get_model_size_bytes(model)

    #torch.initialize(f"{args.net}_{args.b}",model_size)

    if args.c:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.to(device)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare dataset
    train_texts, eval_texts = prepare_data(args.d)
    train_dataset = CustomDataset(train_texts, tokenizer, device=device)
    eval_dataset = CustomDataset(eval_texts, tokenizer, device=device)
    giga = 1024**3
    # Get one batch of training data
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False)
    # input_batch = next(iter(train_loader))
    # if args.gpu:
    #     input_batch = input_batch.cuda()
    # print("Sample batch keys:", batch.keys())
    # print("input_ids shape:", batch['input_ids'].shape)
    time_start = time.time()
    model = get_model_and_a_sample(args, eval_loader, config, args.net)
    sample = next(iter(eval_loader))
    sample = {k: v.to(device) for k, v in sample.items()}
    sample = sample['input_ids']
    model = rotor.Checkpointable(model)
    model.measure(sample)
    model.compute_sequence(mem_limit = args.budget * giga)
    offline_time = time.time() - time_start
    
    # torch.initialize(f"{args.net.replace('/', '_').replace('-', '_')}_{args.b}", model, input_batch)
    
    # if (torch.__version__ == "2.3.0a0"):
    #     print(f"Using customized Pytorch version {torch.__version__}")
    #     torch.initialize(f"{args.net.replace('/', '_').replace('-', '_')}_{args.b}")
    


    time_start = time.time()
    train_loop(model, train_dataset, args.b, args.e)
    time_train = time.time() - time_start
    
    print(f"Max allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"Training completed in {time_train:.2f} seconds")
    print(f"Offline time: {offline_time:.2f} seconds")
    
    
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -rockmate -budget 10
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -no-recompute