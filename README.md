# Reinforcement Learning Molecular Dynamics (rlmc) environment 

`env.py` contains the enviroment for the molecular dynamics simulation.

## Usage

```
env = rlmc_env("5N-spring2D")
env.set_seed(0)
state = env.reset()
simulation_velocity, simulation_positions, reward, done = env.step()
```

## Currently Implemented

simulation type `"5N-spring2D"`

## Python Version

uses `python >= 3.11`