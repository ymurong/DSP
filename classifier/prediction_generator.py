from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from lib.model_selection import fit_model_and_get_predictions, get_train_test_set
import datetime
import pandas as pd
from calendar import monthrange
from lib.mutation_util import daysOfMonth

classifiers_dictionary_0 = {
    'RandomForest': RandomForestClassifier(random_state=0, n_estimators=95, max_depth=9, criterion="log_loss",
                                           max_features="sqrt",
                                           class_weight="balanced_subsample",
                                           n_jobs=-1),
    # 'XGBoost': XGBClassifier(random_state=0, max_depth=2, scale_pos_weight=12.12, eval_metric='mlogloss',
    #                          use_label_encoder=False, n_jobs=-1),
    # 'XGBoostRF': XGBRFClassifier(random_state=0, max_depth=5, scale_pos_weight=12.12,
    #                              eval_metric='mlogloss',
    #                              use_label_encoder=False, n_jobs=-1),
    # 'LightGBM': LGBMClassifier(random_state=0, objective='binary', max_depth=3, scale_pos_weight=12.12, n_jobs=-1)
}

classifiers_dictionary = {
    **classifiers_dictionary_0,
    # 'VotingClassifier': VotingClassifier(estimators=list(classifiers_dictionary_0.items()),
    #                                      voting='soft', weights=[12, 1, 1, 6, 1],
    #                                      flatten_transform=True, n_jobs=-1)
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


def extract_train_test(start_date_training, delta_train, delta_delay, delta_test,
                       final_feature_path="../feature-engineering/final_features.csv"):
    # extract trainset final features
    df_features = pd.read_csv(final_feature_path)
    df_features["tx_datetime"] = pd.to_datetime(df_features["tx_datetime"])
    df_train, df_test = get_train_test_set(
        df_features,
        start_date_training,
        delta_train,
        delta_delay,
        delta_test,
    )
    return df_train, df_test


def train_test_monthly_generator(test_start_month, test_end_month, year=2021):
    """generate train/test data set by month"""
    delay_duration = 1
    train_duration = 3
    test_duration = 1

    for test_month in range(test_start_month, test_end_month + 1):
        train_start_month = test_month - train_duration - delay_duration
        delay_start_month = test_month - delay_duration

        start_date_training = datetime.datetime(year, train_start_month, 1)
        delta_train = daysOfMonth(train_start_month, train_start_month + train_duration - 1)
        delta_delay = daysOfMonth(delay_start_month, delay_start_month + delay_duration - 1)
        delta_test = daysOfMonth(test_month, test_month + test_duration - 1)

        df_train, df_test = extract_train_test(
            start_date_training=start_date_training,
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_test=delta_test
        )
        yield df_train, df_test


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


if __name__ == '__main__':

    classifier = "RandomForest"
    df_pred_prob_all = pd.DataFrame(columns=['psp_reference', 'predict_proba', 'created_at', 'updated_at'])
    for df_train, df_test in train_test_monthly_generator(test_start_month=7, test_end_month=12, year=2021):
        fitted_models_and_predictions_dictionary = fit_predict(df_train, df_test, input_features)
        y_predict_proba = pd.Series(fitted_models_and_predictions_dictionary[classifier]["predictions_test"],
                                    name="predict_proba")
        df_pred_prob = pd.concat([df_test[["psp_reference"]].reset_index(drop=True), y_predict_proba], axis=1)
        df_pred_prob["created_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
        df_pred_prob["updated_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
        df_pred_prob_all = pd.concat([df_pred_prob_all, df_pred_prob], ignore_index=True, axis=0)

    # make sure no duplicated transactions predictions results generated
    assert not (df_pred_prob_all["psp_reference"].duplicated().any())
    df_pred_prob_all.to_csv("../backend/predictions_dump.csv", index=False)
