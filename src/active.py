import json
import logging
import os

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import Subset, SequentialSampler, DataLoader
from src.data import processors
from src import setup, train, sample, cluster

logger = logging.getLogger(__name__)


def cluster_method(sampling):
    """Given the [sampling] method for active learning, return clustering function [f]
     and [condition], boolean that indicates whether sampling
    is conditioned on prior iterations"""
    if "KM" in sampling:
        f = cluster.kmeans
        condition = False
    elif "KP" in sampling:
        f = cluster.kmeans_pp
        condition = True
    elif "FF" in sampling:
        f = cluster.kcenter
        condition = True
    elif sampling == "badge":
        f = cluster.badge
        condition = False
    elif sampling == "alps":
        f = cluster.kmeans
        condition = False
    else:
        #  [sampling] is not cluster-based strategy
        f = None
        condition = None
    return f, condition

def acquire(pool, sampled, args, model, tokenizer):
    """Sample data from unlabeled data [pool].
    The sampling method may need [args], [model], [tokenizer], or previously
    [sampled] data."""
    scores_or_vectors = sample.get_scores_or_vectors(pool, args, model, tokenizer)
    clustering, condition = cluster_method(args.sampling)
    unsampled = np.delete(torch.arange(len(pool)), sampled)
    if clustering is not None:
        # cluster-based sampling method like BADGE and ALPS
        vectors = normalize(scores_or_vectors)
        centers = sampled.tolist()
        if not condition:
            # do not condition on previously chosen points
            queries_unsampled = clustering(
                vectors[unsampled], k = args.query_size
            )
            # add new samples to previously sampled list
            queries = centers + (unsampled[queries_unsampled]).tolist()
        else:
            queries = clustering(
                vectors,
                k = args.query_size,
                centers = centers
            )
        queries = torch.LongTensor(queries)
    else:
        # scoring-based methods like maximum entropy
        scores = scores_or_vectors
        _, queries_unsampled = torch.topk(scores[unsampled], args.query_size)
        queries = torch.cat((sampled, unsampled[queries_unsampled]))
    assert len(queries) == len(queries.unique()), "Duplicates found in sampling"
    assert len(queries) > 0, "Sampling method sampled no queries."
    return queries

def main():
    args = setup.get_args()
    setup.set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # first, get already sampled points
    sampled_file = os.path.join(args.model_name_or_path, 'sampled.pt')
    if os.path.isfile(sampled_file):
        sampled = torch.load(sampled_file)
    else:
        sampled = torch.LongTensor([])

    # decide which model to load based on sampling method
    args.head = sample.sampling_to_head(args.sampling)
    if args.head == "lm":
        # load pre-trained model
        args.model_name_or_path = args.base_model
    model, tokenizer, _, _= setup.load_model(args)


    dataset = train.load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

    logger.info(f"Already sampled {len(sampled)} examples")
    sampled = acquire(dataset, sampled, args, model, tokenizer)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.save(sampled, os.path.join(args.output_dir, 'sampled.pt'))
    logger.info(f"Sampled {len(sampled)} examples")
    return len(sampled)

if __name__ == "__main__":
    main()
