import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def correlation_matrix_plot(X_train):
    corr_matrix = X_train.corr()
    fig, ax = plt.subplots(figsize=(15, 15))  # Sample figsize in inches
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)


def plot_confusion_matrix(y_true, y_pred, h=15, w=15):
    fig, ax = plt.subplots(figsize=(h, w))  # Sample figsize in inches
    sns.set(font_scale=4)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, linewidths=.5, fmt='g', ax=ax)
    plt.show()
