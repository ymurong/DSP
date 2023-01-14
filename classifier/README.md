# Classifier Directory

This directory is used to train and tune the model in order to produce pre-trained model and feed to backend.

* Feature Dataset: [final_features.ipynb](../feature-engineering/final_features.ipynb) is used to produce the final train csv
  file final_features.csv

## Benchmark
* Training: [model_benchmark.py](./model_benchmark.py) is used to benchmark different models (xgboost, random forest, etc).

## XGboosting

* Tuning: [xgboosting_tuning_hyperopt.py](model_tuning/xgboosting_tuning_hyperopt.py) is used to find the best parameters for
  XGboosting. The automatic tuning would take approximately half an hour.
* Training: [xgboosting_fit.py](./xgboosting_fit.py) is used to train the model.

After training, the model would be saved under the **backend/src/resources/pretrained_models** directory called *
*xgboost_classifier_model.pkl**

## XGboost Random Forest

* Tuning: [xgboosting_rf_tuning_hyperopt.py](model_tuning/xgboosting_rf_tuning_hyperopt.py) is used to find the best parameters for
  XGboosting. The automatic tuning would take approximately half an hour.


