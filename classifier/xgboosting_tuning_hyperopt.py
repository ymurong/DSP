import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold

start_date = "2021-01-01"
end_date = "2021-11-30"

# extract trainset final features
df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")
X_train = df_features.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_features["has_fraudulent_dispute"]

# subset of features to fit
X_train_subset = pd.concat(
    [X_train.loc(axis=1)["ip_node_degree", "card_node_degree", "email_node_degree"],
     X_train.loc(axis=1)[["is_credit"]],
     X_train.loc(axis=1)["ip_address_woe":"card_number_woe"],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"]],
    axis=1)
print(f"Train data size: {X_train_subset.shape[0]}")

# extract testset to predict
df_test = pd.read_csv("test_dataset_december.csv")
X_test = df_test[X_train_subset.columns]
y_test = df_test["has_fraudulent_dispute"]
print(f"Test data size: {X_test.shape[0]}")

# use pos_scale_weight to handle unbalanced dataset
scale_pos_weight = y_train.value_counts().to_dict()[False] / y_train.value_counts().to_dict()[True]
print(f"Neg/Pos Ratio: {y_train.value_counts().to_dict()}")
print(f"pos_scale_weight: {scale_pos_weight}")


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
        'max_delta_step': hp.quniform("max_delta_step", 3, 10, 1),
        'gamma': hp.uniform('gamma', 0.01, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'reg_alpha': hp.quniform('reg_alpha', 1, 5, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0.05, 9),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.4, 0.01),
        'scale_pos_weight': hp.uniform('scale_pos_weight', scale_pos_weight * 0.95, scale_pos_weight * 1.2)
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
