#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=200M
#SBATCH --output=curriculum_run.out

module load anaconda
srun python run_curriculum.py
