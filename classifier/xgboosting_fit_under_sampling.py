import numpy as np
import pandas as pd
from lib.sampling import subsampling
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, precision_score
from xgboost import XGBClassifier, plot_importance
import seaborn as sn
import matplotlib.pyplot as plt
from lib.plot_util import correlation_matrix_plot
import pickle

start_date = "2021-01-01"
end_date = "2021-11-30"

df_features = pd.read_csv(f"../feature-engineering/final_features_{end_date}.csv")

# Subsample non_fraudulent transactions records so we have balanced dataset
df_fraudulent = df_features[df_features['has_fraudulent_dispute'] == True]
df_non_fraudulent = df_features[df_features['has_fraudulent_dispute'] == False]
subsample_index = subsampling(df_non_fraudulent.index, len(df_fraudulent))
df_non_fraudulent_subsample = df_non_fraudulent.loc[subsample_index, :]
df_sample = pd.concat([df_non_fraudulent_subsample, df_fraudulent], axis=0)

X_train = df_sample.drop(["date", "psp_reference", "has_fraudulent_dispute", "is_refused_by_adyen"], axis=1)
y_train = df_sample["has_fraudulent_dispute"]
X_train_subset = pd.concat(
    [X_train.loc(axis=1)["ip_node_degree", "card_node_degree", "email_node_degree"],
     X_train.loc(axis=1)[["is_credit"]],
     X_train.loc(axis=1)["ip_address_woe":"card_number_woe"],
     X_train.loc(axis=1)["no_ip", "no_email", "same_country", "merchant_Merchant B", "merchant_Merchant C",
     "merchant_Merchant D", "merchant_Merchant E", "card_scheme_MasterCard", "card_scheme_Other", "card_scheme_Visa",
     "device_type_Linux", "device_type_MacOS", "device_type_Other", "device_type_Windows", "device_type_iOS", "shopper_interaction_POS"]],
    axis=1)

correlation_matrix_plot(X_train_subset)
print(f"Train data size: {X_train_subset.shape[0]}")

df_test = pd.read_csv("test_dataset_december.csv")
X_test = df_test[X_train_subset.columns]
y_test = df_test["has_fraudulent_dispute"]

print(X_train_subset.columns)


def metrics_sklearn(y_valid, y_pred_):
    """模型对验证集和测试集结果的评分"""
    # 准确率
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    # 精准率
    precision = precision_score(y_valid, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    # 召回率
    recall = recall_score(y_valid, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    # F1值
    f1 = f1_score(y_valid, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    # auc曲线下面积
    auc = roc_auc_score(y_valid, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    # ks值
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))


def feature_importance_selected(clf_model):
    """模型特征重要性提取与保存"""
    # 模型特征重要性打印和保存
    feature_importance = clf_model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['特征名称', '重要性'])
    feature_ipt.to_csv('feature_importance.csv', index=False)
    print('特征重要性:', feature_importance)

    # 模型特征重要性绘图
    fig, ax = plt.subplots(figsize=(13, 10))
    plot_importance(clf_model, ax=ax)
    plt.show()


def model_fit():
    """模型训练"""
    # XGBoost训练过程，下面的参数是调试出来的最佳参数组合
    model = XGBClassifier(n_estimators=10, reg_alpha=6, reg_lambda=7)
    model.fit(X_train_subset, y_train)

    # 对验证集进行预测——类别
    y_pred = model.predict(X_test)
    y_test_ = y_test.values
    print('y_test：', y_test_)
    print('y_pred：', y_pred)

    # 对验证集进行预测——概率
    y_pred_proba = model.predict_proba(X_test)
    # 结果类别是1的概率
    y_pred_proba_ = []
    for i in y_pred_proba.tolist():
        y_pred_proba_.append(i[1])
    print('y_pred_proba：', y_pred_proba_)

    # 模型对验证集预测结果评分
    metrics_sklearn(y_test_, y_pred)

    # 模型特征重要性提取、展示和保存
    feature_importance_selected(model)

    return model


def model_save_type(clf_model):
    # 模型训练完成后做持久化，模型保存为model模式，便于调用预测
    pickle.dump(clf_model, open("../backend/src/resources/pretrained_models/xgboost_classifier_model.pkl", "wb"))

    # 模型保存为文本格式，便于分析、优化和提供可解释性
    # clf = clf_model.get_booster()
    # clf.dump_model('./output_models/dump.txt')


if __name__ == '__main__':
    """
        模型训练、评分与保存
        结论：训练集k折交叉验证带来的模型评分提升，未必会在测试集上得到提升
    """
    # 模型训练
    model_xgbclf = model_fit()
    # 模型保存：model和txt两种格式
    model_save_type(model_xgbclf)
