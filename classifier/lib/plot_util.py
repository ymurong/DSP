import seaborn as sns
import matplotlib.pyplot as plt


def correlation_matrix_plot(X_train):
    corr_matrix = X_train.corr()
    fig, ax = plt.subplots(figsize=(15, 15))  # Sample figsize in inches
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
