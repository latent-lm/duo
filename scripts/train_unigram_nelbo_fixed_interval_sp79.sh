#!/bin/bash
#SBATCH -J unigram-nelbo-fi79          # Job name
#SBATCH -o watch_folder/%x_%j.out      # output file (%j expands to jobID)
#SBATCH -N 1                           # Total number of nodes requested
#SBATCH --get-user-env                 # retrieve the user's login environment
#SBATCH --mem=32000                    # memory requested per node
#SBATCH -t 04:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                   # Single GPU
#SBATCH --open-mode=append             # Do not overwrite logs
#SBATCH --requeue                      # Requeue upon pre-emption

set -euo pipefail

cd "$(dirname "$0")/.."

python -u unigram_test.py \
  --proposal fixed_interval \
  --max-steps 79 \
  --lr 1e-4 \
  --loss nelbo \
  --bridge-plot-path unigram_bridge_nelbo_lr1e-4_sp79_new4.jpg \
  --plot-path unigram_nelbo_lr1e-4_sp79_new4.jpg
