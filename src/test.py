import os
import torch
import logging
import argparse
from src import train, setup
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm, trange
import numpy as np
from src.data import compute_metrics


def run_test(model_dir):
    results = {}
    train_args = torch.load(os.path.join(model_dir, 'training_args.bin'))
    train_args.model_name_or_path = model_dir
    model, tokenizer, _, _ = setup.load_model(train_args)
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if train_args.task_name == "mnli" else (train_args.task_name,)
    for eval_task in eval_task_names:
        eval_dataset = train.load_and_cache_examples(train_args, eval_task, tokenizer, evaluate=True, test=True)

        train_args.eval_batch_size = train_args.per_gpu_eval_batch_size * max(1, train_args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=train_args.eval_batch_size)

        # multi-gpu eval
        if train_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(train_args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if train_args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if train_args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if train_args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif train_args.output_mode == "regression":
            preds = np.squeeze(preds)
        np.seterr(divide='ignore', invalid='ignore')
        result = compute_metrics(eval_task, preds, out_label_ids)
        np.seterr(divide='warn', invalid='warn')
        results.update(result)
        with open(os.path.join(model_dir, 'test_results.txt'), 'w') as f:
            for key in sorted(result.keys()):
                f.write("%s = %s\n" % (key, str(result[key])))

    return results

def output_results(model_dir, results, output_csv):
    train_args = torch.load(os.path.join(model_dir, 'training_args.bin'))
    dirname = os.path.basename(os.path.normpath(model_dir))
    if '_' in dirname:
        ranking, size = dirname.split('_')
    else:
        ranking = dirname
        size = ''

    with open(output_csv, 'a') as f:
        for metric in results:
            line = (f'{results[metric]:.4f},{metric},{ranking},{size},'
                    f'{train_args.seed},{train_args.task_name},{train_args.model_type}\n')
            f.write(line)

def is_trained(model_dir, files):
    return ((not 'checkpoint' in model_dir) and ('eval_results.txt' in files) )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    default=None,
    type=str,
    required =True,
    help="Directory of models to be evaluated."
)
args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


for subdir, dirs, files in os.walk(args.models):
    if is_trained(subdir, files) and 'test_results.txt' not in files:
        # run evaluation
        logger.info(f'Testing {subdir}')
        results = run_test(subdir)
