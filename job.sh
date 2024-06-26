#!/bin/bash
#SBATCH --job-name=ml_dyn
#SBATCH --output=logs/output.%j.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a40:1

cd $HOME/repos/mpc_tests
echo "Starting job"

source $HOME/ml/bin/activate # activate the virtual environment

#start tensorboard
srun python ml_dynamics.py
echo "Job finished"

# note: check that the logs directory exists