import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score
from xgboost import XGBClassifier, plot_importance
import pickle

from classifier.lib.plot_util import correlation_matrix_plot, plot_confusion_matrix

start_date = "2021-01-01"
end_date = "2021-11-30"

# extract trainset final features
df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")
X_train = df_features.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_features["has_fraudulent_dispute"]

# subset of features to fit
X_train_subset = pd.concat(
    [X_train.loc(axis=1)["ip_node_degree", "card_node_degree", "email_node_degree"],
     X_train.loc(axis=1)[["is_credit"]],
     X_train.loc(axis=1)["ip_address_woe":"card_number_woe"],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"]],
    axis=1)
print(f"Train data size: {X_train_subset.shape[0]}")
correlation_matrix_plot(X_train_subset)

# extract testset to predict
df_test = pd.read_csv("test_dataset_december.csv")
X_test = df_test[X_train_subset.columns]
y_test = df_test["has_fraudulent_dispute"]
print(f"Test data size: {X_test.shape[0]}")


def metrics_sklearn(y_valid, y_pred_):
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


def feature_importance_selected(clf_model):
    """features importances output"""
    feature_importance = clf_model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    feature_ipt.to_csv('feature_importance.csv', index=False)
    print('feature importances:', feature_importance)

    plot_importance(clf_model)
    plt.show()


def model_fit():
    """model train and test"""
    # XGBoost训练过程，下面的参数是调试出来的最佳参数组合
    model = XGBClassifier(n_estimators=67,
                          max_depth=3,
                          reg_alpha=3,
                          reg_lambda=8.292142504289128,
                          subsample=0.6080786704726493,
                          min_child_weight=1,
                          max_delta_step=5,
                          learning_rate=0.14,
                          gamma=0.46478462240273716,
                          colsample_bytree=0.5265527216427881,
                          scale_pos_weight=14)
    model.fit(X_train_subset, y_train)

    # run model prediction to test set
    y_pred = model.predict(X_test)
    y_test_ = y_test.values

    # extract prediction probability
    y_pred_proba = model.predict_proba(X_test)

    # probability of being positive
    y_pred_proba_ = []
    for i in y_pred_proba.tolist():
        y_pred_proba_.append(i[1])

    # calculate model metrics
    metrics_sklearn(y_test_, y_pred)

    # plot confusion matrix
    plot_confusion_matrix(y_test_, y_pred)

    # model features importance extraction
    feature_importance_selected(model)

    return model


def model_save_type(clf_model):
    """persistence of model after training"""
    pickle.dump(clf_model, open("../backend/src/resources/pretrained_models/xgboost_classifier_model.pkl", "wb"))

    # 模型保存为文本格式，便于分析、优化和提供可解释性
    # clf = clf_model.get_booster()
    # clf.dump_model('./output_models/dump.txt')


if __name__ == '__main__':
    # 模型训练
    model_xgbclf = model_fit()
    # 模型保存：model和txt两种格式
    model_save_type(model_xgbclf)
