from collections import Counter
import math
import scipy.stats as ss
import pandas as pd
import numpy as np


def conditional_entropy(x, y):
    # entropy of x given y
    # in our usecase, y would be the predictor and x would be the target variable
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theil_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, ex = ss.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), p


def calculate_woe_iv(dataset):
    """
    calculate weight of evidence and information value (entrophy based)
    :param dataset:
    :return:
    """
    df = dataset.copy()
    df['pos_rate'] = (df['pos'] + 1) / df['pos'].sum()  # Calculate the proportion of responses (Y=1) within each grouping, plus 1 to prevent the numerator denominator from being 0 when calculating woe
    df['neg_rate'] = (df['neg'] + 1) / df['neg'].sum()  # Calculate the percentage of non-responses (Y=0) within each grouping
    df['woe'] = np.log(df['pos_rate'] / df['neg_rate'])  # Calculate the WOE for each grouping
    df['iv'] = (df['pos_rate'] - df['neg_rate']) * df['woe']  # Calculate the IV for each grouping the higher the better
    iv = df['iv'].sum()
    return iv, df
