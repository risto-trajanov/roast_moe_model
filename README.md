# Switch Transformer - ROAST

## Setup

### Environment

```
conda env create -n roast python=3.9
```

### Pytorch + CUDA

Install pytorch for cuda 11.6

```
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### Requirements

```
pip install -r requirements.txt
```

## Data

For the purpose of this study, we initiated training on a similar dataset, [Code Search Net](https://huggingface.co/datasets/code_search_net), that is consistent with code from various different languages. Due to computing limitations, the training dataset consisted of 2K examples of the given dataset.


## Training

```
export CUDA_VISIBLE_DEVICES=0
python train_switch.py
python train_switch_roast.py
```
