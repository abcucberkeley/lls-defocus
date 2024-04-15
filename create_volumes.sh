#!/bin/bash

HANDLER=slurm
ENV=python
TIMELIMIT='24:00:00'
CPUS=1
MEM='128G'
JOB='create_vol'

if [ $HANDLER = 'slurm' ];then
    j="${ENV} create_volumes.py"

    task="/usr/bin/sbatch"
    task="${task} --qos=abc_normal"
    task="${task} --partition=abc"
    task="${task} --gres=gpu:1"
    task="${task} --constraint='titan'"
    task="${task} --mem='${MEM}'"
    task="${task} --job-name=${JOB}"
    task="${task} --time=${TIMELIMIT}"
    task="${task} --wrap=\"${j}\""


    echo $task | bash
    echo "ABC : Running[$(squeue -u $USER -h -t running -r -p abc | wc -l)], Pending[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
else
    echo $j | bash
fi