from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence


class Hparam_Switch():
    def __init__(self):
        self.model_name_or_path = 'google/switch-base-8'
        self.model_type = 'switchtransformers'
        self.tokenizer_name = 'google/switch-base-8'
        self.cache_dir = '/scratch0/rt52/cached'
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.num_train_epochs = 1
        self.trn_bs = 32
        self.val_bs = 32
        self.overwrite_cache = False
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.n_tpu_cores = 0
        self.n_gpu = 2
        self.max_grad_norm = 1.0
        self.do_train = True
        self.do_predict = True
        self.gradient_accumulation_steps = 4
        self.seed = 42
        self.block_size = 256

class MethodDataset(Dataset):
    """
        This will be superseded by a framework-agnostic approach
        soon.
    """

    def __init__(
        self, tokenizer, df, mode, block_size: int = 256, overwrite_cache=False, local_rank=-1,
    ):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.examples = []
        for lang, code in tqdm(list(zip(df.language.values, df.code.values))):
            inpt = f'{code}'
            outpt = f'<{lang}> {code}'
            input_encodings = tokenizer.encode_plus(inpt, pad_to_max_length = True, max_length = block_size, truncation = True)
            target_encodings = tokenizer.encode_plus(outpt, pad_to_max_length = True, max_length = block_size, truncation = True)

            encodings = {
                'input_ids': torch.tensor(input_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long),
                'attention_mask': torch.tensor(input_encodings['attention_mask'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_ids': torch.tensor(target_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_attention_mask': torch.tensor(target_encodings['attention_mask'] + [tokenizer.pad_token_id], dtype = torch.long)
            }

            self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]
    


class C4Dataset(Dataset):
    """
        This will be superseded by a framework-agnostic approach
        soon.
    """

    def __init__(
        self, tokenizer, df, mode, block_size: int = 256, overwrite_cache=False, local_rank=-1,
    ):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.examples = []
        for text in tqdm(list(df.text.values)):
            inpt = f'{text}'
            outpt = f'{text}'
            input_encodings = tokenizer.encode_plus(inpt, pad_to_max_length = True, max_length = block_size, truncation = True)
            target_encodings = tokenizer.encode_plus(outpt, pad_to_max_length = True, max_length = block_size, truncation = True)

            encodings = {
                'input_ids': torch.tensor(input_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long),
                'attention_mask': torch.tensor(input_encodings['attention_mask'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_ids': torch.tensor(target_encodings['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long),
                'target_attention_mask': torch.tensor(target_encodings['attention_mask'] + [tokenizer.pad_token_id], dtype = torch.long)
            }

            self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]
    

# Translated from: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
@dataclass
class DataCollatorForSeq2SeqMaskLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _noise_span_to_unique_sentinel(self, tokens, mask, max_sentinels, sentinel_id):
        sentineled_toks = tokens.clone()
        prev_tok_noise = torch.nn.functional.pad(mask[:-1], [1, 0])

        first_noise_toks = torch.logical_and(mask, ~prev_tok_noise)
        subse_noise_toks = torch.logical_and(mask, prev_tok_noise)

        sentinels = torch.arange(start = sentinel_id, end = sentinel_id - max_sentinels, step = -1)
        sentineled_toks[first_noise_toks] = sentinels[:first_noise_toks.sum().item()]
        return sentineled_toks[~subse_noise_toks]

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability = 0.15, min_span_length = 1, max_span_length = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        import random
        # print(inputs)
        device = inputs.device
        inpts = inputs.clone()
        span_lengths = torch.randint(low = min_span_length, high = max_span_length + 1, size = (inpts.shape[0],), device = device)
        periods = torch.round(span_lengths / mlm_probability)
        offsets = torch.tensor([random.randint(0, period.item()) for period in periods], device = device)
        masks = torch.stack([(torch.arange(start = 0, end = inpts.shape[1]) + offset) % period < span for offset, period, span in zip(offsets, periods, span_lengths)])

        if self.tokenizer._pad_token is not None:
            padding_mask = inpts.eq(self.tokenizer.pad_token_id)
            masks.masked_fill_(padding_mask, value = False)
        num_masks = torch.floor_divide(masks.sum(axis = 1), span_lengths)
        new_inpts = []
        lbls = []
        for inpt, mask in zip(inpts, masks):
            new_inpts.append(
                self._noise_span_to_unique_sentinel(inpt, mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )
            lbls.append(
                self._noise_span_to_unique_sentinel(inpt, ~mask, 100, self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )

        new_inpts = pad_sequence(new_inpts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lbls = pad_sequence(lbls, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return new_inpts, lbls