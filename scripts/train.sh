#!/bin/bash

HANDLER=slurm
#ENV=~/anaconda3/envs/ml/bin/python
ENV=python
TIMELIMIT='24:00:00'
CPUS=1
MEM='20G'
JOB='test-001'

input_path='/clusterfs/nvme/ethan/dataset/no_amplitude_large'
n_epochs=1000
model_path="/clusterfs/nvme/ethan/lls-defocus/models"
experiment_name='test-001'

if [ $HANDLER = 'slurm' ];then
    while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
    do
        sleep 10s
    done

    j="${ENV} train.py"
    j="${j} --input_path ${input_path}"
    j="${j} --n_epochs ${n_epochs}"
    j="${j} --model_path ${model_path}"
    j="${j} --experiment_name ${experiment_name}"

    task="/usr/bin/sbatch"
    task="${task} --qos=abc_normal --nice=1111111111"
    task="${task} --partition=abc"
    task="${task} --cpus-per-task=${CPUS}"
    task="${task} --mem='${MEM}'"
    task="${task} --job-name=${JOB}"
    task="${task} --time=${TIMELIMIT}"
    task="${task} --wrap=\"${j}\""


    echo $task | bash
    echo "ABC : Running[$(squeue -u $USER -h -t running -r -p abc | wc -l)], Pending[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
else
    echo $j | bash
fi