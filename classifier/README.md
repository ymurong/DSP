# Table of Contents
- [Table of Contents](#table-of-contents)
- [Classifier Directory](#classifier-directory)
  - [Input](#input)
  - [Benchmark](#benchmark)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Feature Importance](#feature-importance)
    - [Model Dump](#model-dump)
  - [Tuning](#tuning)


# Classifier Directory

This directory is used to train and tune the model in order to produce pre-trained model and feed to backend.

## Input
* Feature Dataset: [final_features.csv](../feature-engineering/final_features.csv)
* Best Parameters: [optuna_results.ipynb](./model_tuning/optuna_results.ipynb)

## Benchmark

[model_benchmark.py](./model_benchmark.py) is used to benchmark different models. Once executed, it will train the models by using the provided features and hyper parameters, evaluate the model with a variety of metrics, export feature importance image of each model, dump fitted models for further analysis.

### Training

The models includes:
* random forest
* xgboost
* catboost
* xgboost random forest
* lgbm
* Voting classifier

### Evaluation

Evaluation Metrics includes: 
- accuracy  
- precision    
- recall        
- f1       
- auc  
- block_rate  
- fraud_rate  
- conversion_rate  
- average_precision


### Feature Importance
Feature Importance results of each model (except voting) are stored under feature_importance directory.

### Model Dump
Trained models are dumped under pretrained_models directory for further analysis.


## Tuning

Optuna is used to tune the hyperparams of the listed models. [random_forest_optuna_tuning.ipynb](./model_tuning/random_forest_optuna_tuning.ipynb) demonstrates an example of how we tune the model.

Tuning results are store under model_tuning directory. [optuna_results.ipynb](./model_tuning/optuna_results.ipynb) gives a report of the tuning interpretation.