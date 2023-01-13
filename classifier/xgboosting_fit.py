import pandas as pd
from xgboost import XGBClassifier
import pickle
from lib.model_selection import get_train_test_set, metrics_sklearn, scale_data
from lib.plot_util import correlation_matrix_plot, plot_confusion_matrix, feature_importance_selected
import datetime

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

X_test = df_test[X_train.columns]
y_test = df_test["has_fraudulent_dispute"]
print(f"Test data size: {X_test.shape[0]}")

correlation_matrix_plot(X_train)


def model_fit(X_train, y_train):
    """model train and test"""
    model = XGBClassifier(n_estimators=125,
                          # max_depth=1,
                          # reg_alpha=2,
                          # reg_lambda=8.459380860954308,
                          # subsample=0.6175542840963739,
                          # min_child_weight=1,
                          # max_delta_step=6,
                          # learning_rate=0.3,
                          # gamma=0.8265170906861968,
                          # colsample_bytree=0.8177519479936957,
                          scale_pos_weight=14)
    model.fit(X_train, y_train)

    # model features importance extraction
    feature_importance_selected(model, figsize=(30, 30))

    return model


def model_predict(X_test, y_test, model, threshold=0.5):
    y_pred_proba_ = model.predict_proba(X_test.copy())[:, 1]
    y_pred = (y_pred_proba_ >= threshold).astype(bool)

    # calculate model metrics
    metrics_sklearn(y_test.values, y_pred, y_pred_proba_)
    # plot confusion matrix
    plot_confusion_matrix(y_test.values, y_pred)


def model_save_type(clf_model):
    """persistence of model after training"""
    pickle.dump(clf_model, open("../backend/src/resources/pretrained_models/xgboost_classifier_model.pkl", "wb"))


if __name__ == '__main__':
    # scale
    scale = True
    if scale:
        X_train, X_test = scale_data(X_train, X_test)
    # model training
    model_xgbclf = model_fit(X_train, y_train)
    # model prediction
    model_predict(X_test, y_test, model_xgbclf, threshold=0.5)
    # model persistence
    model_save_type(model_xgbclf)
