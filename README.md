# Data-Driven Discovery of High-Performance Polyimides with Enhanced Heat Resistance and Dielectric Properties
***


PI_GA: Data-Driven Discovery of High-Performance Polyimides with Enhanced Heat Resistance and Dielectric Properties
- https://xxxxxx.xxx

## Background
***
This code is the basis of our work submitted to *XXXXXXX*, which aims to reverse engineer polyimides using multi-task learning and genetic algorithms to bring more insights into polymer design. 

## Prerequisites and dependencies
```
$ env.yml
```
## Usage
***
### Raw data of PI structure and properties
The relevant files are kept in './raw_data'

### ST-learning
The relevant files and code for the single-task learning model are in './ST_learning/'

To get extrapolated data, do:
```commandline
$ python extrapolation.py 
```
A command line for feature engineering:
```commandline
$ python feature_engineering.py
$ python dimension_reduction.py
```
The results of every attempt containing metrics are automatically printed in './ST_learning/output/'.

To carry out an optimization of the single-task model, do:
```commandline
$ python hyperparameter_opt.py
```
To train the single-task model, do:
```commandline
$ python train.py
```

### MT-learning
The relevant files and code for the multi-task learning model are in './MT_learning/'

To carry out an optimization of the multi-task model, do:
```commandline
$ python Hyperparameter_opt.py
```
To train the multi-task model, do:
```commandline
$ python multi_task_learning.py
```

### GA
The relevant files and code for the genetic algorithm are in './GA/'
To generate polyimide, do:
```commandline
$ python GeneticAlgorithm.py
```