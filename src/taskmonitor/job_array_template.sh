#!/bin/bash

SRC_DIR=$(pwd) 
JOB_NAME=$1
SCRIPT_YOU_WANT_TO_RUN=$2
ARRAY_INDEX=$3
CLUSTER=$4
ACCOUNT=$5
PARTITION=$6
TIME_LIMIT=$7
NUM_NODES=$8
NUM_TASKS=$9
CPUS_PER_TASK=${10}
MEMORY=${11}
GPU=${12}

# Shift the first n argument for this template script. Pass the rest to the job script.
shift 12 

MAIL_TYPE="NONE"
MAIL_USER="arman.ashkari@utah.edu"

TIMESTAMP=$(date +"%Y-%m-%d-%H:%M:%S")
LOG_DIR_BASE="/scratch/general/vast/u1472216/research/datastore/slurmlog"
OUT_FILE="${LOG_DIR_BASE}/${TIMESTAMP}-%A-%x/%A.%a-%N-%x-out"
ERR_FILE="${LOG_DIR_BASE}/${TIMESTAMP}-%A-%x/%A.%a-%N-%x-err"

echo "Job name: $JOB_NAME"
echo "Script: ${SRC_DIR}/${SCRIPT_YOU_WANT_TO_RUN}"

echo "************************************"
echo "$(cat $SCRIPT_YOU_WANT_TO_RUN)"
echo "************************************"

echo "Array index: $ARRAY_INDEX"
echo "Command line args: $@"

echo "Cluster: $CLUSTER"
echo "Account: $ACCOUNT"
echo "Partition: $PARTITION"
echo "Constraint: $CONSTRAINT"

echo "Time limit: $TIME_LIMIT"
echo "Number of nodes: $NUM_NODES"
echo "Number of tasks: $NUM_TASKS"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Memory: $MEMORY Gigabyte"
echo "GPU: $GPU"
echo "Mail type: $MAIL_TYPE"

JOB_ID=$(sbatch --job-name=$JOB_NAME \
                --clusters=$CLUSTER \
                --account=$ACCOUNT \
                --partition=$PARTITION \
                --constraint=$CONSTRAINT \
                --time=$TIME_LIMIT \
                --nodes=$NUM_NODES \
                --ntasks=$NUM_TASKS \
                --cpus-per-task=$CPUS_PER_TASK \
		        --array=$ARRAY_INDEX \
                --mem="${MEMORY}G" \
                --gres=$GPU \
                --output=$OUT_FILE \
                --error=$ERR_FILE \
                --mail-type=$MAIL_TYPE \
                --mail-user=$MAIL_USER \
                --no-requeue \
                $SCRIPT_YOU_WANT_TO_RUN "$@" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    LOG_DIR_THIS="${LOG_DIR_BASE}/${TIMESTAMP}-${JOB_ID}-${JOB_NAME}"
    mkdir -p $LOG_DIR_THIS
    echo "Job $JOB_ID submitted. Log files will be availabe at $LOG_DIR_THIS"
else
    echo "ERROR! Job submission failed"
fi
