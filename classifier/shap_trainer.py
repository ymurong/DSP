import shap
import pandas as pd
import datetime
from lib.model_selection import get_train_test_set
from lib.mutation_util import daysOfMonth
import dill as pickle
import json

explanability_scores = {
    "IP Risk": 0.5,
    "Email Risk": 0.5,
    "Risk Card Behaviour": 0.5,
    "Risk Card Amount": 0.5,
    "General Evidences": 0.5
}

explainable_categories = {
    "IP Risk": ['ip_address_nb_tx_1day_window', 'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
                'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window', 'ip_address_risk_30day_window'],
    "Email Risk": ['email_address_nb_tx_1day_window', 'email_address_risk_1day_window',
                   'email_address_nb_tx_7day_window', 'email_address_risk_7day_window',
                   'email_address_nb_tx_30day_window', 'email_address_risk_30day_window'],
    "Risk Card Behaviour": ['card_nb_tx_1day_window', 'card_nb_tx_30day_window',
                            'card_nb_tx_7day_window', 'is_weekend', 'diff_tx_time_in_hours',
                            'is_diff_previous_ip_country'],
    "Risk Card Amount": ['card_avg_amount_1day_window', 'card_avg_amount_7day_window',
                         'card_avg_amount_30day_window',
                         'eur_amount'],
    "General Evidences": ['is_credit', 'no_ip', 'no_email', 'same_country', 'merchant_Merchant B',
                          'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
                          'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa', 'ip_country_GR',
                          'ip_country_IT', 'ip_country_NL', 'ip_country_ZW',
                          'issuing_country_GR', 'issuing_country_IT', 'issuing_country_NL',
                          'issuing_country_ZW', 'device_type_Linux', 'device_type_MacOS',
                          'device_type_Other', 'device_type_Windows', 'device_type_iOS',
                          'zip_code_1104', 'zip_code_2039', 'zip_code_3941', 'zip_code_AAD',
                          'zip_code_BB', 'zip_code_BZD', 'zip_code_DB', 'zip_code_DFFF',
                          'zip_code_EB', 'zip_code_EGHA', 'zip_code_FFR', 'zip_code_FGDD',
                          'zip_code_XDED', 'zip_code_XOL', 'zip_code_ZB']
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


def load_pretrained_model(model_path):
    with open(model_path, 'rb') as handle:
        pickled_model = pickle.load(handle)
        return pickled_model


def getFeatureName(feature_array_exp):
    for element in feature_array_exp:
        if element in input_features:
            return element
    return feature_array_exp[0]


def getExplainableGroup(feature_name):
    for key in explainable_categories:
        if feature_name in explainable_categories[key]:
            return key
    return "General Evidences"


def dump_model(explainer, model_path="./pretrained_models/RandomForest_LIME.pickle"):
    with open(model_path, "wb") as handle:
        pickle.dump(explainer, handle)


if __name__ == '__main__':
    threshold = 0.5
    train_duration = 3
    test_duration = 1
    delay_duration = 1

    rf0 = load_pretrained_model(model_path="./pretrained_models/RandomForest.pickle")
    df_train, df_test = extract_train_test(test_start_month=12)
    X_train = df_train[input_features]
    X_test = df_test[input_features]

    explainer = shap.Explainer(rf0)
    shap_values = explainer(X_train)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])
