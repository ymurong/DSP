import datetime
import pandas as pd
from lib.model_selection import get_train_test_set, metrics_sklearn
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold

# extract trainset final features
df_features = pd.read_csv(f"../feature-engineering/final_features.csv")
df_features["tx_datetime"] = pd.to_datetime(df_features["tx_datetime"])
df_train, df_test = get_train_test_set(
    df_features,
    start_date_training=datetime.datetime(2021, 1, 1),
    delta_train=(datetime.datetime(2021, 10, 31) - datetime.datetime(2021, 1,
                                                                     1)).days,
    delta_delay=(datetime.datetime(2021, 11, 30) - datetime.datetime(2021, 11,
                                                                     1)).days,
    delta_test=(datetime.datetime(2021, 12, 31) - datetime.datetime(2021, 12,
                                                                    1)).days
)

columns = ['is_credit', 'no_ip', 'no_email', 'same_country', 'merchant_Merchant B',
           'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
           'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa',
           'device_type_Linux', 'device_type_MacOS', 'device_type_Other',
           'device_type_Windows', 'device_type_iOS', 'shopper_interaction_POS',
           'is_night', 'is_weekend', 'diff_tx_time_in_hours',
           'is_diff_previous_ip_country', 'card_nb_tx_1day_window',
           'card_avg_amount_1day_window', 'card_nb_tx_7day_window',
           'card_avg_amount_7day_window', 'card_nb_tx_30day_window',
           'card_avg_amount_30day_window', 'email_address_nb_tx_1day_window',
           'email_address_risk_1day_window', 'email_address_nb_tx_7day_window',
           'email_address_risk_7day_window', 'email_address_nb_tx_30day_window',
           'email_address_risk_30day_window', 'ip_address_nb_tx_1day_window',
           'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
           'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window',
           'ip_address_risk_30day_window']

X_train = df_train[columns]
y_train = df_train["has_fraudulent_dispute"]
print(f"Train data size: {X_train.shape[0]}")

# use pos_scale_weight to handle unbalanced dataset
scale_pos_weight = y_train.value_counts().to_dict()[False] / y_train.value_counts().to_dict()[True]
print(f"Neg/Pos Ratio: {y_train.value_counts().to_dict()}")
print(f"pos_scale_weight: {scale_pos_weight}")


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
    clf = xgb.XGBRFClassifier(random_state=_seed, **space)
    score = cross_val_score(clf,
                            X_train,
                            y_train,
                            scoring='average_precision',
                            n_jobs=-1,
                            cv=StratifiedKFold(n_splits=5,
                                               shuffle=True,
                                               random_state=_seed)
                            ).mean(),

    print(f"Average Precision: {score[0]}")

    # # Test Trainset metrics
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_train)
    # metrics_sklearn(y_true=y_train, y_pred=y_pred)

    print(f"param: {space}")
    return {'loss': (1 - score[0]), 'status': STATUS_OK}


def run_optimization():
    trials = Trials()
    best_hyperparameters = fmin(fn=objective,
                                space=xgboost_space(),
                                algo=tpe.suggest,
                                max_evals=50,
                                trials=trials,
                                return_argmin=False)
    print(f"Best params: {best_hyperparameters}")


if __name__ == '__main__':
    run_optimization()
