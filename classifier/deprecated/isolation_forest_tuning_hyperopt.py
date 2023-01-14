import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score, \
    confusion_matrix, average_precision_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from lib.plot_util import correlation_matrix_plot, plot_confusion_matrix

start_date = "2021-01-01"
end_date = "2021-11-30"

# extract trainset final features
df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")
X_train = df_features.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_features["has_fraudulent_dispute"]

# subset of features to fit
X_train_subset = pd.concat(
    [X_train.loc(axis=1)[["is_credit", "eur_amount"]],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"],
     X_train.loc(axis=1)["is_night":"ip_address_risk_30day_window"]],
    axis=1)
print(f"Train data size: {X_train_subset.shape[0]}")
print(X_train_subset.columns)


def isolation_forest_space():
    space = {
        'n_estimators': hp.randint("n_estimators", 50, 500),
        'max_samples': hp.uniform('max_samples', 0.5, 1),
        'bootstrap': True,
        'max_features': hp.uniform('max_features', 0.5, 1),
        # 'contamination': hp.choice("contamination", ['auto', 0.0001, 0.0002])
    }
    return space


def objective(space):
    _seed = 42
    anomalyclassifier = IsolationForest(random_state=_seed, **space)
    score = cross_val_score(anomalyclassifier,
                            X_train_subset,
                            np.array([1 if y == 0 else -1 for y in y_train]),
                            scoring='f1_weighted',
                            n_jobs=-1,
                            cv=StratifiedKFold(n_splits=5,
                                               shuffle=True,
                                               random_state=_seed)
                            ).mean(),

    print(f"F1 Weighted Score: {score[0]}")

    # # Test Trainset metrics
    # y_inlier = anomalyclassifier.fit_predict(X_train_subset, y_train) # 1 for inliers, -1 for outliers.
    # y_inlier[y_inlier == 1] = 0
    # y_inlier[y_inlier == -1] = 1
    # y_pred_proba = -anomalyclassifier.score_samples(X_train_subset)
    #
    # metrics_sklearn(y_train, y_inlier, y_pred_proba)

    print(f"param: {space}")
    return {'loss': (1 - score[0]), 'status': STATUS_OK}


def metrics_sklearn(y_valid, y_pred_, y_pred_proba):
    """model metrics output"""
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


def run_optimization():
    trials = Trials()
    best_hyperparameters = fmin(fn=objective,
                                space=isolation_forest_space(),
                                algo=tpe.suggest,
                                max_evals=50,
                                trials=trials,
                                return_argmin=False)
    print(f"Best params: {best_hyperparameters}")


if __name__ == '__main__':
    run_optimization()
