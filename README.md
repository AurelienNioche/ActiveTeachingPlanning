# ActiveTeachingPlanning

## Data

User data: https://zenodo.org/record/5536917.

## A2C implementation

A2C implementation adapted from https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

## Generate human-like agents

First, you need to estimate population parameters from the data:
    
    python inference_run.py

This will create the file `data/param_exp_data.csv` 
that contains estimates of the population parameters.

Then, 