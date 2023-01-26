# Feature Engineering Directory

The feature-engineerning directory is used to do experiments and produce features.

* baseline_features: some transactions related transformation features

| Category                | Short-Term                                                                                             | Middle-Term                                  | Long-Term                                                       |
|-------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------------------|
| IP Risk                 | total_nb_transactions_1d, total_nb_frauds_1d                                                           | total_nb_transactions_7d, total_nb_frauds_7d | total_nb_transactions_30d, total_nb_frauds_30d                  | 
| Email Risk              | total_nb_transactions_1d, total_nb_frauds_1d                                                           | total_nb_transactions_7d, total_nb_frauds_7d | total_nb_transactions_30d, total_nb_frauds_30d                  |
| Card Risk               | total_nb_transactions_1d, is_night, diff_tx_time_in_hours, ip_country_changed, issuing_ip_same_country | total_nb_transactions_7d, Is_weekend,        | total_nb_transactions_30d                                       |
| Card Amount             | card_avg_amount_1d                                                                                     | card_avg_amount_7d                           | card_avg_amount_30d                                             |
| General Characteristics |                                                                                                        |                                              | is_credit, merchant_dummy, card_scheme_dummy, device_type_dummy |

* categorical_dummies:  one-hot encoding features
  of ['merchant', 'card_scheme', 'is_credit', 'ip_country', 'issuing_country', 'device_type',
  'shopper_interaction', 'zip_code', 'card_bin', 'no_ip', 'no_email', 'same_country']. 
   
* 'ip_country', 'issuing_country' and zip_code** not to be used due to fairness issue.

  
All features would be combined and merged into final features. Training should have include feature selection process instead of including all the features.