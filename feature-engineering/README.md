# Feature Engineering Directory

The feature-engineerning directory is used to do experiments and produce features.

* graph features: we treat every transaction's ip, bank account number and online purchase email as a node on the graph,
  then connect them (if a transaction uses the IP: A, account B, then connect AB).
* woe features: weight of evidence encoding features with ip, email and card number.
* categorical_dummies:  one-hot encoding features
  of ['merchant', 'card_scheme', 'is_credit', 'ip_country', 'issuing_country', 'device_type',
  'shopper_interaction', 'zip_code', 'card_bin', 'no_ip', 'no_email', 'same_country']. **'ip_country', 'issuing_country'
  and zip_code** not to be used due to fairness issue.

All features would be combined and merged into final features. Training should have include feature selection process instead of including all the features.