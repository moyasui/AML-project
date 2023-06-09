# AdvancedMachineLearning
Repository for FYS5429 project: Solving chaotic system with RNN - Spring 2023
This repository contains the source code, selected trained models and results for the project. 
The source code is contained in [**src**](https://github.com/Daniel-Haas-B/AdvancedMachineLearning/tree/main/src).
The plots are in [**Analysis/figs**](https://github.com/Daniel-Haas-B/AdvancedMachineLearning/tree/main/src/Analysis/figs).

## Prerequisites
Python libraries:
- numpy, 
- matplotlib, 
- seaborn, 
- tensorflow

## Data generation
Compilation: 
```
make compile_lorenz
```

Run example:
```
make run_lorenz particles=1000 tf=8 delta_t=0.01
```
- Parameters 
  - particles: number of different initial trajectories;
  - tf: final time;
  - delta_t: time difference between integration steps;

There are pre-generated data in [**csvs**](https://github.com/Daniel-Haas-B/AdvancedMachineLearning/tree/main/src/csvs)

## Reproduce Results
To produce the results for:
- the spiral
```sh
python3 spiral_rnn.py
```
- Lorenz attractor 
with normal RNN:
```sh
python3 lorenz_rnn.py
```
with RNN with physics informed:
```sh
python3 lorenz_rnn_PI.py
```
(optional) FFNN:
```sh
python3 lorenz_FFN.py
```

