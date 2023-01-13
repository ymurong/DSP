import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score, \
    confusion_matrix, average_precision_score

from lib.plot_util import correlation_matrix_plot, plot_confusion_matrix

start_date = "2021-01-01"
end_date = "2021-11-30"

# extract trainset final features
df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")
X_train = df_features.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_features["has_fraudulent_dispute"]

# subset of features to fit
X_train_subset = pd.concat(
    [X_train.loc(axis=1)[["is_credit"]],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"],
     X_train.loc(axis=1)["is_night":"ip_address_risk_30day_window"]],
    axis=1)
print(f"Train data size: {X_train_subset.shape[0]}")
print(X_train_subset.columns)
correlation_matrix_plot(X_train_subset)

# extract testset to predict
df_test = pd.read_csv("../../backend/src/resources/test_dataset_december.csv")
X_test = df_test[X_train_subset.columns]
y_test = df_test["has_fraudulent_dispute"]
print(f"Test data size: {X_test.shape[0]}")


def metrics_sklearn(y_valid, y_pred_, y_pred_proba):
    """model metrics output"""
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    precision = precision_score(y_valid, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    recall = recall_score(y_valid, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    f1 = f1_score(y_valid, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    auc = roc_auc_score(y_valid, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))

    block_rate = len(y_pred_[y_pred_ == True]) / len(y_pred_)
    print('block_rate：%.2f%%' % (block_rate * 100))

    false_neg = confusion_matrix(y_valid, y_pred_)[1, 0]
    fraud_rate = false_neg / len(y_pred_[y_pred_ == False])
    print('fraud_rate：%.2f%%' % (fraud_rate * 100))

    conversion_rate = 1 - block_rate
    print('conversion_rate：%.2f%%' % (conversion_rate * 100))

    average_precision = average_precision_score(y_valid, y_pred_proba)
    print('average_precision：%.2f%%' % (average_precision * 100))


def model_fit():
    anomalyclassifier = IsolationForest(random_state=44,
                                        n_estimators=310,
                                        max_samples=0.9038267154735551,
                                        max_features=0.5199685896451771,
                                        bootstrap=True
                                        ).fit(X_train_subset, y_train)
    y_inlier = anomalyclassifier.predict(X_test)  # 1 for inliers, -1 for outliers.
    y_inlier[y_inlier == 1] = 0
    y_inlier[y_inlier == -1] = 1

    y_pred_proba = -anomalyclassifier.score_samples(X_test)

    metrics_sklearn(y_test, y_inlier, y_pred_proba)

    return anomalyclassifier


def save_outlier_prob(anomalyclassifier):
    y_pred_proba = -anomalyclassifier.score_samples(X_test)
    outlier_prob = pd.concat([df_test["psp_reference"], pd.Series(y_pred_proba, name="outlier_prob")], axis=1)
    outlier_prob.to_csv("test_outlier_prob.csv", index=False)


if __name__ == '__main__':
    model_isof = model_fit()
    save_outlier_prob(model_isof)
