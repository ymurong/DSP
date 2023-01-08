import numpy as np
import pandas as pd
from lib.sampling import subsampling
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from lib.plot_util import correlation_matrix_plot

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


def xgboost_parameters():
    """模型调参过程"""
    # 第一步：确定迭代次数 n_estimators
    # 参数的最佳取值：{'n_estimators': 10}
    # 最佳模型得分:0.9744889348123922
    # params = {'n_estimators': [5, 10, 20, 50, 75, 100, 200]}

    # 第二步：min_child_weight[default=1],range: [0,∞] 和 max_depth[default=6],range: [0,∞]
    # min_child_weight:如果树分区步骤导致叶节点的实例权重之和小于min_child_weight,那么构建过程将放弃进一步的分区,最小子权重越大,算法就越保守
    # max_depth:树的最大深度,增加该值将使模型更复杂,更可能过度拟合,0表示深度没有限制
    # 参数的最佳取值：{'max_depth': 7, 'min_child_weight': 1}
    # 最佳模型得分: 0.9734127986890103
    # params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]}

    # 第三步:gamma[default=0, alias: min_split_loss],range: [0,∞]
    # gamma:在树的叶子节点上进行进一步分区所需的最小损失下降,gamma越大,算法就越保守
    # 参数的最佳取值：{'gamma': 0.5}
    # 最佳模型得分: 0.9717944619156785 无提高使用默认
    # params = {'gamma': [0.01, 0.05, 0.1,  0.2, 0.3, 0.4, 0.5, 0.6]}

    # 第四步：subsample[default=1],range: (0,1] 和 colsample_bytree[default=1],range: (0,1]
    # subsample:训练实例的子样本比率。将其设置为0.5意味着XGBoost将在种植树木之前随机抽样一半的训练数据。这将防止过度安装。每一次提升迭代中都会进行一次子采样。
    # colsample_bytree:用于列的子采样的参数,用来控制每颗树随机采样的列数的占比。有利于满足多样性要求,避免过拟合
    # 参数的最佳取值：{'colsample_bytree': 0.7, 'subsample': 0.8}
    # 最佳模型得分: 0.9742447899557423 无提高使用默认
    # params = {'subsample': [0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}

    # 第五步：alpha[default=0, alias: reg_alpha], 和 lambda[default=1, alias: reg_lambda]
    # alpha:L1关于权重的正则化项。增加该值将使模型更加保守
    # lambda:关于权重的L2正则化项。增加该值将使模型更加保守
    # 参数的最佳取值：{'alpha': 2, 'lambda': 4}
    # 最佳模型得分:  0.9749843464636669
    # params = {'alpha': [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8], 'lambda': [0.05, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    # 第六步：learning_rate[default=0.3, alias: eta],range: [0,1]
    # learning_rate:一般这时候要调小学习率来测试,学习率越小训练速度越慢,模型可靠性越高,但并非越小越好
    # 参数的最佳取值：{'learning_rate': 0.3}
    # 最佳模型得分: 0.9707269892173774  无提高使用默认
    params = {'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.23, 0.25, 0.28, 0.3, 0.4]}

    # 其他参数设置，每次调参将确定的参数加入
    # fine_params = {'n_estimators': 50, 'max_depth': 2, 'min_child_weight': 1, 'gamma': 0.1, 'colsample_bytree': 1,
    #                'subsample': 1, 'reg_alpha': 0.01, 'reg_lambda': 3, 'learning_rate': 0.3}
    fine_params = {'n_estimators': 10, 'alpha': 6, 'lambda': 7}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    """模型调参"""
    # 模型基本参数
    model = XGBClassifier(**other_params)
    # sklearn提供的调参工具，训练集k折交叉验证
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=-1)
    # 模型训练
    optimized_param.fit(X_train_subset, y_train)
    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))

    # 模型参数调整得分变化曲线绘制
    parameters_score = pd.DataFrame(params, means)
    parameters_score['means_score'] = parameters_score.index
    parameters_score = parameters_score.reset_index(drop=True)
    # parameters_score.to_excel('parameters_score.xlsx', index=False)
    # 画图
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(parameters_score.iloc[:, :-1], 'o-')
    plt.legend(parameters_score.columns.to_list()[:-1], loc='upper left')
    plt.title('Parameters_size', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.subplot(2, 1, 2)
    plt.plot(parameters_score.iloc[:, -1], 'r+-')
    plt.legend(parameters_score.columns.to_list()[-1:], loc='upper left')
    plt.title('Score', loc='left', fontsize='xx-large', fontweight='heavy')
    plt.show()


if __name__ == '__main__':
    """
        模型调参
        调参策略：网格搜索、随机搜索、启发式搜索
        补充：此处采用启发式搜索，逐个或逐类参数调整，避免所有参数一起调整导致模型训练复杂度过高
    """
    # xgboost参数组合
    adj_params, fixed_params = xgboost_parameters()
    # 模型调参
    model_adjust_parameters(adj_params, fixed_params)
