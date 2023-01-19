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

OUTPUT_FEATURE = "has_fraudulent_dispute"

TEST_DATA_PATH = "src/resources/test_dataset_december.csv"

XGBOOST_MODEL_PATH = "../resources/pretrained_models/XGBoost.pickle"
RF_MODEL_PATH = "../resources/pretrained_models/RandomForest.pickle"

EXPLAINABLE_CATEGORIES = {
    "IP Risk": ['ip_address_nb_tx_1day_window', 'ip_address_risk_1day_window','ip_address_nb_tx_7day_window', 'ip_address_risk_7day_window','ip_address_nb_tx_30day_window', 'ip_address_risk_30day_window'],
    "Email Risk": ['email_address_nb_tx_1day_window', 'email_address_risk_1day_window','email_address_nb_tx_7day_window', 'email_address_risk_7day_window','email_address_nb_tx_30day_window', 'email_address_risk_30day_window'],
    "Risk Card Behaviour": ['card_nb_tx_1day_window', 'card_avg_amount_1day_window', 'card_nb_tx_7day_window', 'is_weekend', 'diff_tx_time_in_hours', 'is_diff_previous_ip_country'],
    "Risk Card Amount": ['card_avg_amount_7day_window','card_nb_tx_30day_window', 'card_avg_amount_30day_window', 'eur_amount'],
    "Circumstancial Evidences": ['is_credit', 'no_ip', 'no_email', 'same_country', 'merchant_Merchant B','merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E','card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa','ip_country_GR', 'ip_country_IT', 'ip_country_NL', 'ip_country_ZW',
       'issuing_country_GR', 'issuing_country_IT', 'issuing_country_NL',
       'issuing_country_ZW', 'device_type_Linux', 'device_type_MacOS',
       'device_type_Other', 'device_type_Windows', 'device_type_iOS',
       'zip_code_1104', 'zip_code_2039', 'zip_code_3941', 'zip_code_AAD',
       'zip_code_BB', 'zip_code_BZD', 'zip_code_DB', 'zip_code_DFFF',
       'zip_code_EB', 'zip_code_EGHA', 'zip_code_FFR', 'zip_code_FGDD',
       'zip_code_XDED', 'zip_code_XOL', 'zip_code_ZB']
}