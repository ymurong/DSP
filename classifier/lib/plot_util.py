import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from xgboost import plot_importance


def correlation_matrix_plot(X_train):
    corr_matrix = X_train.corr()
    fig, ax = plt.subplots(figsize=(15, 15))  # Sample figsize in inches
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)


def plot_confusion_matrix(y_true, y_pred, h=15, w=15):
    fig, ax = plt.subplots(figsize=(h, w))  # Sample figsize in inches
    sns.set(font_scale=4)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, linewidths=.5, fmt='g', ax=ax)
    plt.show()


def feature_importance_selected(clf_model, figsize=(100, 30)):
    """features importances output"""
    feature_importance = clf_model.get_booster().get_fscore()
    feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_ipt = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    feature_ipt.to_csv('feature_importance.csv', index=False)
    print('feature importances:', feature_importance)
    fig, ax = plt.subplots(figsize=figsize)
    plot_importance(clf_model, ax=ax)
    plt.show()