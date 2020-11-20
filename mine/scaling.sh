#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=jupyter-notebook
#SBATCH --output=jupyter-notebook-%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexn@minerva.kgi.edu
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:4

/usr/bin/time python -u scaling_mi.py

