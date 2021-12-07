# DoubleML

Using the Python package **DoubleML**, an implementation of the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).

Documentation and website: [https://docs.doubleml.org/](https://docs.doubleml.org/)

## Installation requirements

- Python
- sklearn
- numpy
- scipy
- pandas
- statsmodels
- joblib
- seaborn
- doubleml

### Repository DoubleML simulations

    .
    ├── simulation/                  
		 ├── simulation_features.py  # script to run initial simulation of coefficient estimation
		 ├── ThetaDistributionPlots/ # initial simulation of estimators comparison with gaussian data OLS. vs DML
		 ├── Bernoulli/             # comparison of gaussian features vs bernoulli features
			 ├── CI_simulation.py   # script to run the simulation of gaussian vs bernoulli
			 ├── CIplots/ 				# the visualisation of results of simulation
		 └── README.md
