#!/bin/bash

# change if needed
SEED=125
TASK=imdb
# end

TASK_MODELS=models/$SEED/$TASK
OUTPUT=analysis/${TASK}.csv

python -m src.analyze --task_models $TASK_MODELS --output $OUTPUT
