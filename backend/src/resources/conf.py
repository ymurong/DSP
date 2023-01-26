INPUT_FEATURES = ['is_credit', 'same_country', 'merchant_Merchant B',
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

EXPLAINABLE_CATEGORIES = {
    "ip_risk": ['ip_address_nb_tx_1day_window', 'ip_address_risk_1day_window', 'ip_address_nb_tx_7day_window',
                'ip_address_risk_7day_window', 'ip_address_nb_tx_30day_window', 'ip_address_risk_30day_window'],
    "email_risk": ['email_address_nb_tx_1day_window', 'email_address_risk_1day_window',
                   'email_address_nb_tx_7day_window', 'email_address_risk_7day_window',
                   'email_address_nb_tx_30day_window', 'email_address_risk_30day_window'],
    "risk_card_behaviour": ['card_nb_tx_1day_window', 'card_nb_tx_30day_window',
                            'card_nb_tx_7day_window', 'is_weekend', 'is_night', 'diff_tx_time_in_hours',
                            'is_diff_previous_ip_country', 'same_country'],
    "risk_card_amount": ['card_avg_amount_1day_window', 'card_avg_amount_7day_window',
                         'card_avg_amount_30day_window'],
    "general_evidences": ['is_credit', 'merchant_Merchant B',
                          'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
                          'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa',
                          'device_type_Linux', 'device_type_MacOS', 'device_type_Other',
                          'device_type_Windows', 'device_type_iOS', 'shopper_interaction_POS']
}

OUTPUT_FEATURE = "has_fraudulent_dispute"
SENSITIVE_FEATURE = "issuing_country"

TEST_DATA_PATH = "../resources/test_dataset_december.csv"
XGBOOST_MODEL_PATH = "../resources/pretrained_models/XGBoost.pickle"
RF_MODEL_PATH = "../resources/pretrained_models/RandomForest.pickle"
RF_EXPLAINER_PATH = "../resources/pretrained_models/RandomForest_LIME.pickle"
