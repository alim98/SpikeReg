#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/almik/SpikeReg5/SpikeReg/final_logs/final_oasis.%j.out
#SBATCH -e /u/almik/SpikeReg5/SpikeReg/final_logs/final_oasis.%j.err
# Initial working directory:
#SBATCH -D /u/almik/SpikeReg5/SpikeReg
# Job Name:
#SBATCH -J final_oasis
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#
#SBATCH --mem=128000  # Memory per node (in MB)
#
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:2 
#
#SBATCH --mail-type=none
#SBATCH --mail-user=ali.Mikaeili@brain.mpg.de
# Wall clock limit (max. is 24 hours):
#SBATCH --time=24:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
module purge
module load intel/21.2.0 impi/2021.2

# Activate the local Python environment
source /u/almik/synapseClusterEMLight/synapse_env_new/bin/activate

# Run the program:

# python examples/train_oasis.py \
#     --data-root /u/almik/SpikeReg2/data\
#     --multi-gpu \
#     --gpu-ids "0,1" \
#     --num-workers 8 \
#     --batch-size 8 \
#     --patch-size 32 \
#     --device cuda \
#     --checkpoint-dir /u/almik/SpikeReg5/checkpoints/oasis/oasis \
#     --log-dir logs/oasis \
#     --finetune-epochs 50 \
#     --skip-pretrain \
#     --resume /u/almik/SpikeReg5/checkpoints/oasis/oasis/converted_model.pth

# python examples/train_oasis.py --data-root /u/almik/SpikeReg2/data --num-workers 8 --batch-size 8 --patch-size 32 --device cuda --checkpoint-dir /u/almik/SpikeReg5/checkpoints/final_oasis --log-dir logs/final_oasis --pre --finetune-epochs 50 
python examples/train_oasis.py \
    --data-root /u/almik/SpikeReg2/data \
    --multi-gpu \
    --gpu-ids "0,1" \
    --num-workers 8 \
    --pretrain-epochs 10 \
    --finetune-epochs 10 \
    --batch-size 8 \
    --patch-size 32 \
    --device cuda \
    --checkpoint-dir /u/almik/SpikeReg5/SpikeReg/checkpoints/final_oasis \
    --log-dir /u/almik/SpikeReg5/SpikeReg/logs/final_oasis