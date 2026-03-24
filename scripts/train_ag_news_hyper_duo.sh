#!/bin/bash
#SBATCH -J hyper-duo-ag-news            # Job name
#SBATCH -o watch_folder/%x_%j.out        # output file (%j expands to jobID)
#SBATCH -N 1                             # Total number of nodes requested
#SBATCH --get-user-env                   # retrieve the users login environment
#SBATCH --mem=64000                      # server memory requested (per node)
#SBATCH -t 960:00:00                     # Time limit (hh:mm:ss)
#SBATCH --partition=sun                  # Request partition
#SBATCH --nodelist=sun-compute-03
#SBATCH --constraint="a6000"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # Single A6000 GPU
#SBATCH --open-mode=append               # Do not overwrite logs
#SBATCH --requeue                        # Requeue upon pre-emption

# Hyperbolic DUO on ag_news, single A6000
# Two forward passes per step (variance reduction), so halve batch vs DUO
python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=ag_news \
  wandb.name=hyper-duo-ag-news \
  model=small \
  model.n_blocks=8 \
  model.n_heads=8 \
  algo=hyperbolic_duo \
  algo.trainset_path=/share/thickstun/sychou/workspace/research/duo/hyper_dataset/word_embedding/trainset_gpt2_ag_news_ml256_bs32.pt \
  model.length=256
