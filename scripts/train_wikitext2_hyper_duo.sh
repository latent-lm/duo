#!/bin/bash
#SBATCH -J hyper-duo-wikitext2            # Job name
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

# Hyperbolic DUO on wikitext2, single A6000
# Two forward passes per step (variance reduction), so halve batch vs DUO
python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=wikitext2 \
  wandb.name=hyper-duo-wikitext2 \
  model=small \
  model.n_blocks=8 \
  model.n_heads=8 \
  algo=hyperbolic_duo \
  model.length=256
