#!/bin/bash
#SBATCH --job-name=ml_dyn
#SBATCH --output=logs/output.%j.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:50:00
#SBATCH --gres=gpu:a40:1
cd $HOME/repos/mpc_tests
echo "Starting job"

#start tensorboard
srun singularity exec --nv $HOME/slurm_singularity_cluster/mycontainer.sif python ml_dynamics2.py
echo "Job finished"

# note: check that the logs directory exists