# Repository DoubleML simulations

The initial simulations are using synthetic data to 

The scripts used to generate results are:

- simulation_features.py produces the plots held in ThetaDistributionPlots folder. 
- CI_simulation.py Inside Bernoulli folder produces the plots in the CI plots folder. 

The distribution plots are created using the following parameters
```
Simulations: 100
Number of features: 100
C = 0.5 #hardcoded inference

Data generation:
X = Random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.
D = random binomial with prob:0.5
Weight coefficients: Random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.
Noise = 0
Y = X.Weights + c*D

RF parameters:
n_estimators: 100
max_depth=100

DML parameters:
Folds = 2
```
