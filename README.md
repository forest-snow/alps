# ALPS
Code repository for EMNLP 2020 proceedings paper "Cold-start Active Learning through Self-supervised Language Modeling".  The main contribution of the paper is an active learning algorithm called ALPS (Active Learning through Processing Surprisal) that is based on the language modeling objective.  

# Installation
1. Create virtual environment with Python 3.7+
2. Run following commands:
```
git clone https://github.com/forest-snow/alps.git
cd alps
pip install -r requirements.txt
```
# Organization
The repository is organized as the following subfolders:

1. `src`: source code
2. `scripts`: scripts for running experiments
3. `data`: folder for datasets
4. `models`: saved models from running experiments
5. `analysis`: analysis of active learning experiments

# Usage
All commands below should be ran in the top-level directory `alps`.

## Fine-tune model on full training dataset
To simply fine-tune a model on the full training dataset, run `bash scripts/train.sh`.  After fine-tuning, this model will be saved under a subdirectory called `base` in `models` directory.  Results on dev set will be saved in `eval_results.txt`.

You may modify the parameters (like model type, task, seed, etc.) in `scripts/train.sh`by configuring the variables at the top of the script.  

## Run active learning simulations
To simulate active learning, run `bash scripts/active_train.sh`.  This script will sample data for a fixed number of iterations and then fine-tune the model on the sampled data for each iteration.  The fine-tuned model will be saved under a subdirectory called `{strategy}_{size}` where `strategy` is the active learning strategy used to sample data and `size` is the number of examples used to fine-tune the model.  Results on dev set will be saved in `eval_results.txt`.

To modify parameters in `scripts/active_train.sh`, you can configure the variables at the top of the script.  Please read the instructions below for more information.

### Naming conventions of strategies
Here are the naming conventions of the strategies from the paper:

1. Random sampling: `rand`
2. Max. entropy sampling: `entropy`
3. ALPS: `alps`
4. BADGE: `badge`
5. BERT-KM: `bertKM`
6. FT-BERT-KM: `FTbertKM`

So, whenever you want to use ALPS, you would pass in `alps` as input to the commands presented below.

### No warm-starting required
For active learning strategies that DO NOT require a model already fine-tuned on downstream task (`rand`, `alps`, and `bertKM`), you set variable `SAMPLING`to the strategy's name and variable `COLDSTART` to `none`.  This will use method specified in`SAMPLING` to sample data on each iteration.

### Warm-starting required
For active learning strategies that DO require a model already fine-tuned on downstream task (`rand`, `alps`, and `bertKM`), you set variable `SAMPLING`to the strategy's name and variable `COLDSTART` to the method used for sampling data in the first iteration.  For instance, max. entropy sampling would have `SAMPLING` set to `entropy` and `COLDSTART` set to `rand`.  

**NOTE:** you must run simulation for method specified in `COLDSTART` for at least one iteration. For example, run `rand`for 1 iteration before running simulations for`entropy`.

### Sample size
To set the size of data sampled on each iteration, configure the variable `INCREMENT`.  To set the maximum size of total data sampled, configure the variable `MAX_SIZE`.  The number of iterations would be `MAX_SIZE\INCREMENT`.

## Test fine-tuned models
To test models that have been fine-tuned, run
```
python -m src.test --models models
```
This will iterate through every model located in subdirectories of folder `models` and evaluate them on the test dataset.  However, it will skip over any models that are just checkpoints or were not evaluated on a dev set (models trained with scripts will automatically be tested on dev set).  The script will output results in `test_results.txt`

## Analyze active learning sampled batches
To analyze the uncertainty and diversity of batched sampled with active learning, run `bash scripts/analyze.sh`.

This will output a CSV file in `analysis` folder containing uncertainty and diversity scores for each sampled batch.  The header of the CSV file will be`sampling,iteration,task,diversity,uncertainty`.  Each row indicates the diversity and uncertainty scores for data sampled with strategy at a certain iteration for a task.

# Citation
```
@inproceedings{yuan2020alps,
  title={Cold-start Active Learning through Self-supervised Language Modeling},
  author={Yuan, Michelle and Lin, Hsuan-Tien and Boyd-Graber, Jordan},
  booktitle={Empirical Methods in Natural Language Processing},
  year={2020}
}
```



