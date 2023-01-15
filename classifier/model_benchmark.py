from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from classifier.lib.model_selection import fit_model_and_get_predictions, get_train_test_set, \
    performance_assessment_model_collection
import datetime
import pandas as pd
import pickle

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
                             use_label_encoder=False, n_jobs=-1),
    'XGBoostRF': XGBRFClassifier(random_state=0, n_estimators=60, max_depth=5, sampling_method="uniform",
                                 booster="dart", scale_pos_weight=12.12,
                                 eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1),
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


def extract_train_test(final_feature_path="../feature-engineering/final_features.csv"):
    # extract trainset final features
    df_features = pd.read_csv(final_feature_path)
    df_features["tx_datetime"] = pd.to_datetime(df_features["tx_datetime"])
    df_train, df_test = get_train_test_set(
        df_features,
        start_date_training=datetime.datetime(2021, 8, 1),
        delta_train=(datetime.datetime(2021, 10, 31) - datetime.datetime(2021, 8,
                                                                         1)).days,
        delta_delay=(datetime.datetime(2021, 11, 30) - datetime.datetime(2021, 11,
                                                                         1)).days,
        delta_test=(datetime.datetime(2021, 12, 31) - datetime.datetime(2021, 12,
                                                                        1)).days
    )
    return df_train, df_test


def fit_predict(df_train, df_test, input_features, output_feature="has_fraudulent_dispute"):
    fitted_models_and_predictions_dictionary = {}
    for classifier_name in classifiers_dictionary:
        model_and_predictions = fit_model_and_get_predictions(classifiers_dictionary[classifier_name], df_train,
                                                              df_test,
                                                              input_features=input_features,
                                                              output_feature=output_feature,
                                                              scale=True)
        fitted_models_and_predictions_dictionary[classifier_name] = model_and_predictions
    return fitted_models_and_predictions_dictionary


def assessment(fitted_models_and_predictions_dictionary, df_test, threshold=0.5):
    performances = performance_assessment_model_collection(fitted_models_and_predictions_dictionary, df_test,
                                                           threshold)
    return performances


def plot_feature_importances(fitted_models_and_predictions_dictionary, nlargest=15, figsize=(20, 10), fontsize=10,
                             export=True, show=False):
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


if __name__ == '__main__':
    threshold = 0.5
    DUMP = True
    adyen_results()
    df_train, df_test = extract_train_test()
    fitted_models_and_predictions_dictionary = fit_predict(df_train, df_test, input_features)
    performances = assessment(fitted_models_and_predictions_dictionary, df_test, threshold=threshold)
    plot_feature_importances(fitted_models_and_predictions_dictionary, export=True, show=False)
    if DUMP:
        dump_models(fitted_models_and_predictions_dictionary)
    print(performances.to_string())
