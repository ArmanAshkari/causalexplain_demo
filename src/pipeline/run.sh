#!/bin/bash

set -e

CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate masterenv

export OMP_NUM_THREADS=1

NUM_PROC=$SLURM_CPUS_PER_TASK
echo "NUM_PROC: $NUM_PROC"

TASK_ID=$1
echo "TASK_ID: $TASK_ID"

PY_MODULE=$2
echo "PY_MODULE: $PY_MODULE"

BATCH_SIZE=$3
echo "BATCH_SIZE: $BATCH_SIZE"

# Consumed 3 arguments
shift 3

START=$(($SLURM_ARRAY_TASK_ID * $BATCH_SIZE))
echo "START: $START"

# ***** CHECK NAME OF THE SCRIPT AND CL ARGS ORDER ****
COMMAND="time python -m $PY_MODULE $TASK_ID $SLURM_ARRAY_JOB_ID $NUM_PROC $START $BATCH_SIZE $@"

echo "$COMMAND"

eval "$COMMAND"