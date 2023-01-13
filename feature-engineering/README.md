# Feature Engineering Directory

The feature-engineerning directory is used to do experiments and produce features.

* baseline_features: some transactions related transformation features
['is_night', 'is_weekend', 'diff_tx_time_in_hours',
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
  

* categorical_dummies:  one-hot encoding features
  of ['merchant', 'card_scheme', 'is_credit', 'ip_country', 'issuing_country', 'device_type',
  'shopper_interaction', 'zip_code', 'card_bin', 'no_ip', 'no_email', 'same_country']. **'ip_country', 'issuing_country'
  and zip_code** not to be used due to fairness issue.

  
All features would be combined and merged into final features. Training should have include feature selection process instead of including all the features.