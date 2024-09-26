#!/bin/bash
#SBATCH --job-name=image_imagenet32_LFM_5blocks_push+distill                        # Job name
#SBATCH --account=gts-yxie77-paid                   # charge account
#SBATCH -N1 --gres=gpu:1 # Number of nodes and GPUs required
#SBATCH --mem-per-cpu=32G                 # Memory per core
#SBATCH -t 1400                               # Duration of the job: integer = mins
#SBATCH -q inferno                               # QOS Name
#SBATCH -o output_%j.txt                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=dwu381@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR

python /storage/home/hcoda1/3/dwu381/scratch/AnnealingFlow_Final_Version/Annealing_Flow.py --AnnealingFlow_config /storage/home/hcoda1/3/dwu381/scratch/AnnealingFlow_Final_Version/truncated.yaml