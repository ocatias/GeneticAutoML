# Genetic Auto ML
_(Implemented for the Machine Learning course at TU Wien)_

We implement a genetic algorithm that can select from different machine learning algorithms and automatically tune their hyperparameters. Our evaluation shows that it gets almost as good results as TPOT and H2O and in some rare cases even slightly beats H2O.  
In its current state it supports  Random Forest (sklearn), XGB, Ridge (sklearn) and Radius Neighbors Classifiers (sklearn). It can easily be adapted to support more classifiers and to also support feature selection.


## Evaluation
Evaluation was done on 4 small datasets:
- [Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone)
- [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- [Tic-Tac-Toe Endgames](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
- [Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition) (due to limited computing resources we only used the first 5000 rows of this dataset)

On each dataset we let all three AutoML algorithms run for an hour each (in total 12h). As a score we use f1-macro (higher is better). The results can be seen here:

<img src="/Pictures/results.png" alt="Results" width="400">

## Interesting Graphs

<img src="/Pictures/aba_algo_score.png" alt="Results" width="400"> <img src="/Pictures/car_algo_score.png" alt="Results" width="400">
<img src="/Pictures/ttt_algo_score.png" alt="Results" width="400"> <img src="/Pictures/car_algo_distribution.png" alt="Results" width="400">

<img src="/Pictures/parallelisation.png" alt="Results" width="400">
