import argparse
import util
from util import *
import pandas as pd
from pynvml import *
import os
import torch
from FakeRoast.FakeRoastUtil_v2 import *
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["small", "full", "code"], default="code")
args = parser.parse_args()

cuda_device = 3

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = f"3,4"

def print_gpu_utilization(cuda_device=3):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(cuda_device)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()

hparams = util.Hparam_Switch()

tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name)
tokenizer.add_special_tokens({'mask_token': '<mask>'})

data_folder = '/scratch0/rt52/'

if args.dataset == "small":
    train_dataset_filename = 'train_dataset_switch_small.pt'
    if 'train_dataset_switch_small.pt' in os.listdir(data_folder):
        train_dataset  = torch.load(data_folder + 'train_dataset_switch_small.pt')
    else:
        train_df_0 = pd.read_csv(data_folder + 'train_c4_0.csv')
        train_df_0 = train_df_0.sample(n=1000, random_state=42)
        torch.save(train_df_0, data_folder + 'train_dataset_switch_small.pt')
    if 'valid_dataset.pt' in os.listdir(data_folder):
        valid_dataset = torch.load(data_folder + 'valid_dataset.pt')
    else:
        val_df_0 = pd.read_csv(data_folder + 'validation_c4.csv')
        valid_dataset = util.C4Dataset(tokenizer, val_df_0, 'valid')
        torch.save(valid_dataset, data_folder + 'valid_dataset.pt')
elif args.dataset == "full":
    train_dataset_filename = 'train_dataset.pt'
    if 'train_dataset.pt' in os.listdir(data_folder):
        train_dataset  = torch.load(data_folder + 'train_dataset.pt')
    else:
        train_df_0 = pd.read_csv(data_folder + 'train_c4_0.csv')
        train_df_1 = pd.read_csv(data_folder + 'train_c4_1.csv')
        trn_df = pd.concat([train_df_0, train_df_1])
        train_dataset = util.C4Dataset(tokenizer, trn_df, 'train')
        torch.save(train_dataset, data_folder + 'train_dataset.pt')
    if 'valid_dataset.pt' in os.listdir(data_folder):
        valid_dataset = torch.load(data_folder + 'valid_dataset.pt')
    else:
        val_df_0 = pd.read_csv(data_folder + 'validation_c4.csv')
        valid_dataset = util.C4Dataset(tokenizer, val_df_0, 'valid')
        torch.save(valid_dataset, data_folder + 'valid_dataset.pt')
elif args.dataset == "code":
    tokenizer.add_tokens(['{', '}', '<java>', '<python>', '<'])
    train_dataset  = torch.load('train_data.pt')
    valid_dataset = torch.load('valid_data.pt')
    
model = SwitchTransformersForConditionalGeneration.from_pretrained(
    hparams.model_name_or_path, cache_dir=hparams.cache_dir
)  # .to('cuda:0'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)

model.to('cuda:0')

# mapper_args = {"mapper": "pareto", "hasher": "uhash", "block_k": 16, "block_n": 16, "block": 8, "seed": 1011}
# roaster = ModelRoasterGradScaler(model_og, True, sparsity=0.5, verbose=None, mapper_args=mapper_args)
# model = roaster.process()
# del model_og
print('After model initialization.')
# print_gpu_utilization()

training_args = TrainingArguments(
    output_dir="./switch_trained/switch_code_data",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=hparams.trn_bs,
    per_device_eval_batch_size=hparams.val_bs,
    gradient_accumulation_steps=hparams.gradient_accumulation_steps,
    logging_strategy = 'epoch',
    save_strategy = 'epoch',
    save_total_limit=2,
    do_train = True,
    do_eval = True,
    do_predict = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=util.DataCollatorForSeq2SeqMaskLanguageModeling(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# %%time
trainer.train()

# %%time
trainer.evaluate()