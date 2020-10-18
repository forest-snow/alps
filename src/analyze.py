import argparse
import torch
from src.data import processors
import os
import json
import numpy as np
from transformers import AutoTokenizer
from src import setup, train
from torch.utils.data import Subset, SequentialSampler, DataLoader
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from pathlib import Path
import pandas as pd

def compute_entropy(sampled, dataset, model, train_args):
    """Compute average entropy in label distribution for examples in [sampled]."""
    all_entropy = None
    data = Subset(dataset, sampled)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=train_args.per_gpu_eval_batch_size)
    for batch in tqdm(dataloader, desc="Computing entropy"):
        batch = tuple(t.to(train_args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if train_args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if train_args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            logits = outputs[0]
            categorical = Categorical(logits = logits)
            entropy = categorical.entropy()
        if all_entropy is None:
            all_entropy = entropy.detach().cpu().numpy()
        else:
            all_entropy = np.append(all_entropy, entropy.detach().cpu().numpy(), axis=0)
    avg_entropy = all_entropy.mean()
    return avg_entropy


def token_set(data, train_args):
    all_tokens = set()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)
    for batch in tqdm(dataloader, desc="Getting tokens"):
        with torch.no_grad():
            tokens = batch[0].unique().tolist()
            all_tokens = all_tokens.union(tokens)
    return all_tokens

def jaccard(a, b):
    ji = len(a.intersection(b))/len(a.union(b))
    return ji

def compute_diversity(sampled, data, train_args):
    # compare jaccard similarity between sampled and unsampled points
    data_sampled = Subset(data, sampled)
    unsampled = np.delete(torch.arange(len(data)), sampled)
    data_unsampled =  Subset(data, unsampled)
    tokens_sampled = token_set(data_sampled, train_args)
    tokens_unsampled = token_set(data_unsampled, train_args)
    ji = jaccard(tokens_sampled, tokens_unsampled)
    return ji

def get_stats(model_path, base_model, dataset):
    sampling, sample_size = model_path.name.split('_')
    sampled = torch.load(model_path / 'sampled.pt')
    diversity = compute_diversity(sampled, dataset, train_args)
    entropy = compute_entropy(sampled, dataset, base_model, train_args)
    stats = {
        "sampling":sampling,
        "iteration":int(sample_size)/100,
        "task":model_path.parent.name,
        "diversity":diversity,
        "uncertainty":entropy
    }
    return stats



TASKNAME = {'sst-2':'SST-2', 'cola':'CoLA'}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_models",
    default=None,
    type=Path,
    help="Directory of models for task"
)
parser.add_argument(
    "--output",
    default=None,
    type=Path,
    help="Path to output file"
)
parser.add_argument(
    "--clear",
    action="store_true",
    help="Flag for clearing csv"
)
args = parser.parse_args()

# get base model trained on full data
train_args = torch.load(args.task_models / 'base' /  'training_args.bin' )
base_model, tokenizer, model_class, tokenizer_class = setup.load_model(train_args)

# load training data
task = train_args.task_name
processor = processors[task]()
if task in TASKNAME:
    task = TASKNAME[task]
dataset = train.load_and_cache_examples(train_args, train_args.task_name, tokenizer, evaluate=False)


# analyze samples from different active learning iterations
all_stats = []
for model_path in args.task_models.glob('**/') :
    print(model_path.name)
    if (model_path / 'sampled.pt').exists():
        stats = get_stats(model_path, base_model, dataset)
        all_stats.append(stats)

# output results
df = pd.DataFrame(all_stats)
if args.clear or not args.output.exists():
    df.to_csv(args.output, index=False, mode='w')
else:
    df.to_csv(args.output, index=False, mode='a', header=False)






