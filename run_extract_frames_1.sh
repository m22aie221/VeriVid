#!/bin/bash
#SBATCH --job-name=opensora_job         # Job name
#SBATCH --partition=fat                 # Choose the appropriate partition
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gres=gpu:2                    # Include 1 GPU for the task
#SBATCH --mem=32gb                      # Total memory limit
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --output=VeriVid%j.log          # Standard output and error log
#SBATCH --mail-type=ALL                 # Send email on job completion, failure, etc.
#SBATCH --mail-user=m22aie221@iitj.ac.in  # Your email address for notifications

# Print some useful info
date; hostname; pwd

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#python custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth

python extract_frames.py --base_dir /scratch/data/m22aie221/workspace/VeriVid

echo "Job completed successfully."
