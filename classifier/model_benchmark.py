import datetime
import os
import pathlib
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from lib.model_selection import fit_model_and_get_predictions, get_train_test_set, \
    performance_assessment_model_collection
from lib.mutation_util import daysOfMonth
from lib.plot_util import plot_feature_importance

classifiers_dictionary_0 = {
    'RandomForest': RandomForestClassifier(random_state=0, n_estimators=95, max_depth=9, criterion="log_loss",
                                           max_features="sqrt",
                                           class_weight="balanced_subsample",
                                           n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=0, n_estimators=9, max_depth=5, objective='binary', scale_pos_weight=12.12,
                               n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=0, n_estimators=6, max_depth=7, sampling_method="uniform", booster="gbtree",
                             scale_pos_weight=12.12, eval_metric='mlogloss',
                             n_jobs=-1),
    'XGBoostRF': XGBRFClassifier(random_state=0, n_estimators=60, max_depth=5, sampling_method="uniform",
                                 booster="dart", scale_pos_weight=12.12,
                                 eval_metric='mlogloss', n_jobs=-1),
    # 'CatBoost': CatBoostClassifier(random_state=0, n_estimators=2, max_depth=6, bootstrap_type="Bernoulli",
    #                                grow_policy="SymmetricTree", silent=True),
}

classifiers_dictionary = {
    **classifiers_dictionary_0,
    'VotingClassifier': VotingClassifier(estimators=list(classifiers_dictionary_0.items()),
                                         voting='soft', weights=[6, 3, 1, 1],
                                         flatten_transform=True, n_jobs=-1)
}

input_features = ['is_credit', 'same_country', 'merchant_Merchant B',
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


# input_features = ['is_credit', 'same_country', 'merchant_Merchant B',
#                   'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
#                   'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa',
#                   'device_type_Linux', 'device_type_MacOS', 'device_type_Other',
#                   'device_type_Windows', 'device_type_iOS', 'shopper_interaction_POS',
#                   'is_night', 'is_weekend', 'diff_tx_time_in_hours',
#                   'is_diff_previous_ip_country', 'card_nb_tx_1day_window',
#                   'card_avg_amount_1day_window', 'card_nb_tx_7day_window',
#                   'card_avg_amount_7day_window', 'card_nb_tx_30day_window',
#                   'card_avg_amount_30day_window', 'email_address_nb_tx_1day_window',
#                   'email_address_risk_1day_window', 'email_address_nb_tx_7day_window',
#                   'email_address_risk_7day_window', 'email_address_nb_tx_30day_window',
#                   'email_address_risk_30day_window', 'ip_address_nb_tx_1day_window',
#                   'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
#                   'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window',
#                   'ip_address_risk_30day_window',
#                   'IPScore', 'IPScore_ST',
#                   'IPScore_MT', 'IPScore_LT', 'EmailScore', 'EmailScore_ST',
#                   'EmailScore_MT', 'EmailScore_LT', 'CHScore', 'CHScore_ST', 'CHScore_MT',
#                   'CHScore_LT', 'MerScore', 'MerScore_ST', 'MerScore_MT', 'MerScore_LT',
#                   'TrxScore', 'TrxScore_ST', 'TrxScore_MT', 'TrxScore_LT']

# input_features = ['is_credit', 'same_country', 'shopper_interaction_POS', 'merchant_Merchant B',
#                   'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
#                   'is_night', 'is_weekend', 'diff_tx_time_in_hours',
#                   'is_diff_previous_ip_country', 'card_nb_tx_1day_window',
#                   'card_avg_amount_1day_window', 'card_nb_tx_7day_window',
#                   'card_avg_amount_7day_window', 'card_nb_tx_30day_window',
#                   'card_avg_amount_30day_window', 'email_address_nb_tx_1day_window',
#                   'email_address_risk_1day_window', 'email_address_nb_tx_7day_window',
#                   'email_address_risk_7day_window', 'email_address_nb_tx_30day_window',
#                   'email_address_risk_30day_window', 'ip_address_nb_tx_1day_window',
#                   'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
#                   'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window',
#                   'ip_address_risk_30day_window']

# input_features = ['is_credit', 'same_country', 'shopper_interaction_POS', 'merchant_Merchant B',
#                   'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
#                   'is_night', 'is_weekend', 'diff_tx_time_in_hours',
#                   'is_diff_previous_ip_country', 'card_nb_tx_1day_window',
#                   'card_avg_amount_1day_window', 'card_nb_tx_7day_window',
#                   'card_avg_amount_7day_window', 'card_nb_tx_30day_window',
#                   'card_avg_amount_30day_window', 'email_address_nb_tx_1day_window',
#                   'email_address_risk_1day_window', 'email_address_nb_tx_7day_window',
#                   'email_address_risk_7day_window', 'email_address_nb_tx_30day_window',
#                   'email_address_risk_30day_window', 'ip_address_nb_tx_1day_window',
#                   'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
#                   'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window',
#                   'ip_address_risk_30day_window',
#                   'IPScore', 'IPScore_ST',
#                   'IPScore_MT', 'IPScore_LT', 'EmailScore', 'EmailScore_ST',
#                   'EmailScore_MT', 'EmailScore_LT', 'CHScore', 'CHScore_ST', 'CHScore_MT',
#                   'CHScore_LT',
#                   'TrxScore', 'TrxScore_ST', 'TrxScore_MT', 'TrxScore_LT']


def extract_train_test(test_start_month, final_feature_path="../feature-engineering/final_features.csv"):
    """
    Given the test_start_month, return the df_train and df_test needed.
    Ex: if test_start_month = 8, then based on the train_duration, test_duration, delay_duration, the function automatically
    extract the df_train and df_test needed -> train period would be 4,5,6 and delay period would be 7. delay period need to
    be ignored during training to simulate that we won't know immediately know whether it is fraudulent transaction or not.
    delta_train -> days of training period
    delta_delay -> days of delay period
    delta_test> days of testing period
    """

    delay_start_month = test_start_month - delay_duration
    train_start_month = test_start_month - delay_duration - train_duration

    df_features = pd.read_csv(final_feature_path)
    df_features["tx_datetime"] = pd.to_datetime(df_features["tx_datetime"])

    start_date_training = datetime.datetime(2021, train_start_month, 1)
    delta_train = daysOfMonth(train_start_month, train_start_month + train_duration - 1)
    delta_delay = daysOfMonth(delay_start_month, delay_start_month + delay_duration - 1)
    delta_test = daysOfMonth(test_start_month, test_start_month + test_duration - 1)

    df_train, df_test = get_train_test_set(
        df_features,
        start_date_training=start_date_training,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_test=delta_test
    )
    return df_train, df_test


def fit_predict(df_train, df_test, input_features, output_feature="has_fraudulent_dispute"):
    fitted_models_and_predictions_dictionary = {}
    for classifier_name in classifiers_dictionary:
        model_and_predictions = fit_model_and_get_predictions(classifiers_dictionary[classifier_name], df_train,
                                                              df_test,
                                                              input_features=input_features,
                                                              output_feature=output_feature,
                                                              scale=False)
        fitted_models_and_predictions_dictionary[classifier_name] = model_and_predictions
    return fitted_models_and_predictions_dictionary


def assessment(fitted_models_and_predictions_dictionary, df_test, threshold=0.5):
    performances = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, df_test,
                                                           threshold)
    return performances


def plot_feature_importances(fitted_models_and_predictions_dictionary, nlargest=25, figsize=(20, 10), fontsize=10,
                             export=True, show=False):
    generate_feature_importances_csv(fitted_models_and_predictions_dictionary)
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        if classifier_name != "VotingClassifier":
            plot_feature_importance(model_and_predictions['classifier'], input_features, nlargest=nlargest,
                                    figsize=figsize,
                                    fontsize=fontsize, export=export, show=show)


def dump_models(fitted_models_and_predictions_dictionary):
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        with open(f"./pretrained_models/{classifier_name}.pickle", 'wb') as handle:
            pickle.dump(model_and_predictions['classifier'], handle)


def adyen_results(final_feature_path="../feature-engineering/final_features.csv"):
    df_features = pd.read_csv(final_feature_path)
    adyen_results = pd.crosstab(df_features["has_fraudulent_dispute"], df_features["is_refused_by_adyen"])
    print(adyen_results)


def evaluate(test_start_month, only_print=True):
    df_train, df_test = extract_train_test(test_start_month)
    fitted_models_and_predictions_dictionary = fit_predict(df_train, df_test, input_features)
    performances = assessment(fitted_models_and_predictions_dictionary, df_test, threshold=threshold)
    plot_feature_importances(fitted_models_and_predictions_dictionary, export=True, show=False)
    train_start_month = test_start_month - delay_duration - train_duration

    report(f">>>>>>  Training months: {train_start_month} -> {train_start_month + train_duration - 1}\n", only_print)
    report(f">>>>>>  Testing months: {test_start_month} -> {test_start_month + test_duration - 1}\n", only_print)
    report(performances.to_string() + "\n")
    return fitted_models_and_predictions_dictionary


def cross_validation(test_start_month_marker, test_end_month_marker, report_path="cross_val_score.txt"):
    file = pathlib.Path(report_path)
    if file.exists():
        os.remove(report_path)
    for test_start_month in range(test_start_month_marker, test_end_month_marker + 1):
        evaluate(test_start_month, only_print=False)


def generate_feature_importances_csv(fitted_models_and_predictions_dictionary):
    feature_importance_all = []
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        if classifier_name != "VotingClassifier":
            feat_importance_one = pd.DataFrame([model_and_predictions['classifier'].feature_importances_],
                                               columns=input_features)
            feat_importance_one.index = [classifier_name]
            feature_importance_all.append(feat_importance_one)
    df_feature_importances = pd.concat(feature_importance_all, axis=0)
    print(">>>>>>>> Feature Importances")
    print(df_feature_importances.to_string())
    df_feature_importances.to_csv("./feature_importance/feature_importance.csv", index=True)


def report(text, report_path="cross_val_score.txt", only_print=True):
    if only_print:
        print(text)
        return
    file = pathlib.Path(report_path)
    if file.exists():
        with open(file, "a") as handle:
            handle.write(text)
    else:
        with open(file, "w") as handle:
            handle.write(text)


if __name__ == '__main__':
    threshold = 0.5
    train_duration = 1
    test_duration = 1
    delay_duration = 1
    # adyen_results()

    # dump model
    fitted_models_and_predictions_dictionary = evaluate(test_start_month=12)
    dump_models(fitted_models_and_predictions_dictionary)

    # generate cross_validation results
    # cross_validation(test_start_month_marker=5, test_end_month_marker=12)

# Without Graph
# >>>>>>  Training months: 8 -> 10
# >>>>>>  Testing months: 12 -> 12
#
#                   accuracy  precision    recall        f1       auc  block_rate  fraud_rate  conversion_rate  average_precision
# RandomForest      0.949532   0.649161  0.892678  0.751689  0.923765    0.117672    0.010409         0.882328           0.650694
# LightGBM          0.948245   0.648120  0.864594  0.740868  0.910333    0.114153    0.013080         0.885847           0.676316
# XGBoost           0.944812   0.621399  0.908726  0.738086  0.928457    0.125139    0.008928         0.874861           0.657899
# XGBoostRF         0.947043   0.637681  0.882648  0.740429  0.917859    0.118445    0.011391         0.881555           0.664476
# VotingClassifier  0.949532   0.648944  0.893681  0.751899  0.924220    0.117844    0.010313         0.882156           0.660406


# With Graph
# >>>>>>  Training months: 8 -> 10
# >>>>>>  Testing months: 12 -> 12
#
#                   accuracy  precision    recall        f1       auc  block_rate  fraud_rate  conversion_rate  average_precision
# RandomForest      0.949618   0.650293  0.889669  0.751377  0.922449    0.117071    0.010693         0.882929           0.652405
# LightGBM          0.948245   0.648120  0.864594  0.740868  0.910333    0.114153    0.013080         0.885847           0.676196
# XGBoost           0.945069   0.622849  0.907723  0.738776  0.928144    0.124710    0.009021         0.875290           0.658736
# XGBoostRF         0.947043   0.637881  0.881645  0.740211  0.917404    0.118273    0.011486         0.881727           0.661362
# VotingClassifier  0.949876   0.651061  0.892678  0.752961  0.923953    0.117329    0.010405         0.882671           0.661868


# Without Graph
# >>>>>>  Training months: 7 -> 9
# >>>>>>  Testing months: 11 -> 11
#
#                   accuracy  precision    recall        f1       auc  block_rate  fraud_rate  conversion_rate  average_precision
# RandomForest      0.950989   0.641172  0.868798  0.737828  0.913437    0.107562    0.011670         0.892438           0.637158
# LightGBM          0.949414   0.639289  0.832415  0.723180  0.895958    0.103361    0.014837         0.896639           0.626053
# XGBoost           0.945388   0.605048  0.898567  0.723159  0.923996    0.117889    0.009128         0.882111           0.634436
# XGBoostRF         0.950026   0.637031  0.861080  0.732302  0.909388    0.107299    0.012353         0.892701           0.638680
# VotingClassifier  0.950901   0.640194  0.871003  0.737973  0.914397    0.107999    0.011480         0.892001           0.637821


# With Graph
# >>>>>>  Training months: 7 -> 9
# >>>>>>  Testing months: 11 -> 11
#
#                   accuracy  precision    recall        f1       auc  block_rate  fraud_rate  conversion_rate  average_precision
# RandomForest      0.950726   0.640523  0.864388  0.735805  0.911280    0.107124    0.012056         0.892876           0.625889
# LightGBM          0.949326   0.638983  0.831312  0.722568  0.895407    0.103273    0.014933         0.896727           0.623887
# XGBoost           0.945125   0.603397  0.900772  0.722689  0.924861    0.118502    0.008936         0.881498           0.631762
# XGBoostRF         0.946000   0.613815  0.862183  0.717102  0.907705    0.111500    0.012313         0.888500           0.634105
# VotingClassifier  0.950814   0.639225  0.873208  0.738117  0.915357    0.108437    0.011289         0.891563           0.632607
