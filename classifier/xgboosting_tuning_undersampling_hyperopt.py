import numpy as np
import pandas as pd
from lib.sampling import subsampling
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
from lib.plot_util import correlation_matrix_plot
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold

start_date = "2021-01-01"
end_date = "2021-11-30"

df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")

# Subsample non_fraudulent transactions records so we have balanced dataset
df_fraudulent = df_features[df_features['has_fraudulent_dispute'] == True]
df_non_fraudulent = df_features[df_features['has_fraudulent_dispute'] == False]
subsample_index = subsampling(df_non_fraudulent.index, len(df_fraudulent))
df_non_fraudulent_subsample = df_non_fraudulent.loc[subsample_index, :]
df_sample = pd.concat([df_non_fraudulent_subsample, df_fraudulent], axis=0)

X_train = df_sample.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_sample["has_fraudulent_dispute"]
X_train_subset = pd.concat(
    [X_train.loc(axis=1)["ip_node_degree", "card_node_degree", "email_node_degree"],
     X_train.loc(axis=1)[["is_credit"]],
     X_train.loc(axis=1)["ip_address_woe":"card_number_woe"],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"]],
    axis=1)

correlation_matrix_plot(X_train_subset)
print(f"Train data size: {X_train_subset.shape[0]}")

df_test = pd.read_csv("test_dataset_december.csv")
X_test = df_test[X_train_subset.columns]
y_test = df_test["has_fraudulent_dispute"]


def metrics_sklearn(y_true, y_pred):
    """模型对验证集和测试集结果的评分"""
    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    # 精准率
    precision = precision_score(y_true, y_pred)
    print('Precision：%.2f%%' % (precision * 100))

    # 召回率
    recall = recall_score(y_true, y_pred)
    print('Recall：%.2f%%' % (recall * 100))

    # F1值
    f1 = f1_score(y_true, y_pred)
    print('F1：%.2f%%' % (f1 * 100))

    # auc曲线下面积
    auc = roc_auc_score(y_true, y_pred)
    print('AUC：%.2f%%' % (auc * 100))

    # ks值
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))


def xgboost_space():
    space = {
        'n_estimators': hp.randint("n_estimators", 5, 200),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_depth': hp.randint("max_depth", 1, 20),
        'max_delta_step': hp.quniform("max_delta_step", 5, 10, 1),
        'gamma': hp.uniform('gamma', 0.01, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'reg_alpha': hp.quniform('reg_alpha', 1, 5, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0.05, 9),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.4, 0.01),
    }
    return space


def objective(space):
    _seed = 42
    clf = xgb.XGBClassifier(random_state=_seed, **space)
    score = cross_val_score(clf,
                            X_train_subset,
                            y_train,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            cv=StratifiedKFold(n_splits=5,
                                               shuffle=True,
                                               random_state=_seed)
                            ).mean(),

    print(f"F1 Weighted Score: {score[0]}")

    # Test Trainset metrics
    clf.fit(X_train_subset, y_train)
    y_pred = clf.predict(X_train_subset)
    metrics_sklearn(y_true=y_train, y_pred=y_pred)

    print(f"param: {space}")
    return {'loss': (1 - score[0]), 'status': STATUS_OK}


def run_optimization():
    trials = Trials()
    best_hyperparameters = fmin(fn=objective,
                                space=xgboost_space(),
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials,
                                return_argmin=False)
    print(f"Best params: {best_hyperparameters}")


if __name__ == '__main__':
    run_optimization()
