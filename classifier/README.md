# Classifier Directory

This directory is used to train and tune the model in order to produce pre-trained model and feed to backend.

* Feature Dataset: [final_features.ipynb](../feature-engineering/final_features.ipynb) is used to produce the final train csv
  file final_features.csv

## Benchmark
Training/Evaluation/Dump: [model_benchmark.py](./model_benchmark.py) is used to benchmark different models:
* random forest
* xgboost
* catboost
* xgboost random forest
* lgbm

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

Trained models are dumped under pretrained_models directory for further analysis.


## Tuning

Optuna is used to tune the hyperparams of the listed models. [random_forest_optuna_tuning.ipynb](./model_tuning/random_forest_optuna_tuning.ipynb) demonstrates an example of how we tune the model.

Tuning results are store under model_tuning directory. [optuna_results.ipynb](./model_tuning/optuna_results.ipynb) gives a report of the tuning interpretation.