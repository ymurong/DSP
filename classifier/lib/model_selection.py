import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score, \
    confusion_matrix, average_precision_score
import time
import pandas as pd


def get_train_test_set(transactions_df,
                       start_date_training=datetime.datetime(2021, 1, 1),
                       delta_train=(datetime.datetime(2021, 10, 31) - datetime.datetime(2021, 1, 1)).days,
                       delta_delay=(datetime.datetime(2021, 11, 30) - datetime.datetime(2021, 11, 1)).days,
                       delta_test=(datetime.datetime(2021, 12, 31) - datetime.datetime(2021, 12, 1)).days
                       ):
    """
    Given the whole feature dataset, split them into train and test dataset considering a delay period.
    It accounts for the fact that, in a real-world fraud detection system, the label of a transaction (fraudulent or genuine)
    is only known after a customer complaint, or thanks to the result of a fraud investigation.
    :param transactions_df:
    :param start_date_training:
    :param delta_train:
    :param delta_delay:
    :param delta_test:
    :return:
    """
    # Get the training set data
    train_df = transactions_df[(transactions_df.tx_datetime >= start_date_training) & (
            transactions_df.tx_datetime < start_date_training + datetime.timedelta(days=delta_train))]

    # Get the test set data
    test_df = transactions_df[
        (transactions_df.tx_datetime >= start_date_training + datetime.timedelta(days=delta_train + delta_delay)) &
        (transactions_df.tx_datetime < start_date_training + datetime.timedelta(
            days=delta_train + delta_delay + delta_test))]

    # Sort data sets by ascending order of time
    train_df = train_df.sort_values('tx_datetime')
    test_df = test_df.sort_values('tx_datetime')

    return train_df, test_df


def scale_data(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test


def pd_metrics(y_valid, y_pred_, y_pred_proba):
    """transform model metrics into dataframe"""
    accuracy = accuracy_score(y_valid, y_pred_)

    precision = precision_score(y_valid, y_pred_)

    recall = recall_score(y_valid, y_pred_)

    f1 = f1_score(y_valid, y_pred_)

    auc = roc_auc_score(y_valid, y_pred_)

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))

    block_rate = len(y_pred_[y_pred_ == True]) / len(y_pred_)

    false_neg = confusion_matrix(y_valid, y_pred_)[1, 0]
    fraud_rate = false_neg / len(y_pred_[y_pred_ == False])

    conversion_rate = 1 - block_rate

    average_precision = average_precision_score(y_valid, y_pred_proba)

    return pd.DataFrame(
        [[accuracy, precision, recall, f1, auc, block_rate, fraud_rate, conversion_rate, average_precision]],
        columns=["accuracy", "precision", "recall", "f1", "auc", "block_rate", "fraud_rate", "conversion_rate",
                 "average_precision"])


def metrics_sklearn(y_valid, y_pred_, y_pred_proba):
    """print model metrics output"""
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    precision = precision_score(y_valid, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    recall = recall_score(y_valid, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    f1 = f1_score(y_valid, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    auc = roc_auc_score(y_valid, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))

    block_rate = len(y_pred_[y_pred_ == True]) / len(y_pred_)
    print('block_rate：%.2f%%' % (block_rate * 100))

    false_neg = confusion_matrix(y_valid, y_pred_)[1, 0]
    fraud_rate = false_neg / len(y_pred_[y_pred_ == False])
    print('fraud_rate：%.2f%%' % (fraud_rate * 100))

    conversion_rate = 1 - block_rate
    print('conversion_rate：%.2f%%' % (conversion_rate * 100))

    average_precision = average_precision_score(y_valid, y_pred_proba)
    print('average_precision：%.2f%%' % (average_precision * 100))


def fit_model_and_get_predictions(classifier, df_train, df_test,
                                  input_features, output_feature="has_fraudulent_dispute", scale=True):
    """trains a model and returns predictions for a test set"""
    # By default, scales input data
    X_train = df_train[input_features]
    y_train = df_train[output_feature]
    X_test = df_test[input_features]
    y_test = df_test[output_feature]

    if scale:
        X_train, X_test = scale_data(X_train, X_test)

    # We first train the classifier using the `fit` method, and pass as arguments the input and output features
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_execution_time = time.time() - start_time

    # We then get the predictions on the training and test data using the `predict_proba` method
    # The predictions are returned as a numpy array, that provides the probability of fraud for each transaction
    start_time = time.time()
    predictions_test = classifier.predict_proba(X_test)[:, 1]
    prediction_execution_time = time.time() - start_time

    predictions_train = classifier.predict_proba(X_train)[:, 1]

    # The result is returned as a dictionary containing the fitted models,
    # and the predictions on the training and test sets
    model_and_predictions_dictionary = {'classifier': classifier,
                                        'predictions_test': predictions_test,
                                        'predictions_train': predictions_train,
                                        'training_execution_time': training_execution_time,
                                        'prediction_execution_time': prediction_execution_time
                                        }

    return model_and_predictions_dictionary


def performance_assessment_model_collection(fitted_models_and_predictions_dictionary, test_df, threshold=0.5):
    test_df = test_df.copy()

    performances_all = []
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        y_true = test_df["has_fraudulent_dispute"]
        y_pred_proba = model_and_predictions['predictions_test']
        y_pred = (y_pred_proba >= threshold).astype(bool)
        df_performances_one = pd_metrics(y_true, y_pred, y_pred_proba)
        df_performances_one.index = [classifier_name]
        performances_all.append(df_performances_one)

    df_performances = pd.concat(performances_all, axis=0)
    return df_performances
