import numpy as np


def subsampling(sample_index, size):
    sub_sample_index = np.random.choice(sample_index, size, replace=False)
    return sub_sample_index
