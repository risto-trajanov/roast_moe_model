import argparse
import util_roast as util
from util_roast import *
import pandas as pd
from pynvml import *
import gc
import os
import torch
from FakeRoast.FakeRoastUtil_v2 import *
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersConfig,
    TrainingArguments,
    Trainer
)
import code
# os.environ["CUDA_VISIBLE_DEVICES"] = f"0,1,3"

cuda_device = 0
# torch.cuda.set_device(cuda_device)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["small", "full", "code"], default="code")
parser.add_argument("--sparsity", choices=["0.1", "0.2", "0.3", "0.4", "0.5"], default="0.5")
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

sparsity = float(args.sparsity)

def print_gpu_utilization(cuda_device=1):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(cuda_device)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()

hparams = util.Hparam_Switch()

# TOKENIZER INIT

tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name)
tokenizer.add_special_tokens({'mask_token': '<mask>'})

data_folder = '/scratch0/rt52/'

# DATA INIT

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
    
# MODEL INIT

model_og = SwitchTransformersForConditionalGeneration.from_pretrained(
    hparams.model_name_or_path, cache_dir=hparams.cache_dir
).to('cuda:0')

print_gpu_utilization(1)

# model_og_roast_param = ModelRoastableParameters(model_og, module_limit_size=25000)
# model_og.to('cuda:0')

# print_gpu_utilization(1)
# s = model_og_roast_param.process()
# roastable = s['roastable']
# total = s['all']
# fixed = total - roastable

# print("MODEL PARAMETERS: ", get_module_params(model_og))
# # print("ROASTABLE: ", roastable)
# # print("FIXED: ", fixed)

# # Sum the sizes of all model parameters
# total_params = sum(p.numel() for p in model_og.parameters())

# # Convert the total number of parameters to megabytes (MB)
# total_params_mb = total_params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter

# print(f"Total model parameters: {total_params}")
# print(f"Total model size (MB): {total_params_mb:.2f} MB")

# save model

# model name format: roasted_model_{sparsity}.pt, sparsity = 0_1, 0_2, 0_3, 0_4, 0_5

sparsity_str = str(sparsity).replace('.', '_')
roasted_model_name = f'roasted_model_{sparsity_str}.pt'

if roasted_model_name in os.listdir('./roasted_models'):
    print("Loading roasted model.")
    model = torch.load(f'./roasted_models/{roasted_model_name}')
else:
    print("Roasting model.")
    mapper_args = {"mapper": "pareto", "hasher": "uhash", "block_k": 16, "block_n": 16, "block": 8, "seed": 1011}
    roaster = ModelRoasterGradScaler(model_og, True, sparsity=sparsity, verbose=NONE, module_limit_size=25000, mapper_args=mapper_args)
    model = roaster.process()
    torch.save(model, f'./roasted_models/{roasted_model_name}')


# Sum the sizes of all model parameters
total_params = sum(p.numel() for p in model.parameters())

# Convert the total number of parameters to megabytes (MB)
total_params_mb = total_params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter

print(f"Total model parameters: {total_params}")
print(f"Total model size (MB): {total_params_mb:.2f} MB")


af = get_module_params(model)
# print("ROASTED MDOEL THEORETICAL PARAMETERS: ", int(roastable * sparsity))
print("ROASTED MODEL ACTUAL PARAMETERS", af)

del model_og
model_og = None

gc.collect()

torch.cuda.empty_cache()

model.to('cuda:0')

print_gpu_utilization()

print('After model initialization.')

# TRAINING

training_args = TrainingArguments(
    output_dir=f"./switch_roasted_trained/switch_roast_code_data_{sparsity_str}",
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

trainer = util.CustomTrainer(
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