# Classifier Directory

This directory is used to train and tune the model in order to produce pre-trained model and feed to backend.

* Trainset: [final_features.ipynb](../feature-engineering/final_features.ipynb) is used to produce the final train csv
  file final_features_2021-11-30.csv
* Testset: [prepare_test_data.ipynb](./prepare_test_data.ipynb) is used to produce the test csv file
  test_dataset_december.csv

## XGboosting

* Tuning: [xgboosting_tuning_hyperopt.py](./xgboosting_tuning_hyperopt.py) is used to find the best parameters for
  XGboosting. The automatic tuning would take approximately half an hour.
* Training: [xgboosting_fit.py](./xgboosting_fit.py) is used to train the model.

After training, the model would be saved under the **backend/src/resources/pretrained_models** directory called *
*xgboost_classifier_model.pkl**

## Random Forest

* Tuning&Training: [random_forest.ipynb](./random_forest.ipynb) is used to do experiments with random forest such as
  tuning, training and testing.

