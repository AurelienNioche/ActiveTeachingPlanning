#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --mem-per-cpu=200M
#SBATCH --output=curriculum_run.out

module load anaconda
python run_curriculum.py
