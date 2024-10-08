import numpy as np
from scipy.stats import ttest_ind
import h5py
import matplotlib.pyplot as plt

def ttest(a, b, axis=0, equal_var=True, nan_policy='propagate',
          alternative='two.sided'):
    tval, pval = ttest_ind(a=a, b=b, axis=axis, equal_var=equal_var,
                           nan_policy=nan_policy)
    if alternative == 'greater':
        if tval < 0:
            pval = 1 - pval / 2
        else:
            pval = pval / 2
    elif alternative == 'less':
        if tval < 0:
            pval /= 2
        else:
            pval = 1 - pval / 2
    else:
        assert alternative == 'two.sided'
    return tval, pval


def loaddata():
    data_1 = h5py.File('./data/mostdata.mat', 'r')
    out_1 = np.array(data_1['canary ']).reshape(5000, 1)
    data_2 = h5py.File('./data/leastdata.mat', 'r')
    out_2 = np.array(data_2['extra']).reshape(5000, 1)

    return out_1, out_2



if __name__ == "__main__":
    UMI_1, UMI_2 = loaddata()

    tval, pval = ttest(UMI_1, UMI_2, alternative="greater")
    print('tval 0 hypothesis Mostdata <= lestdata: ', tval, ' pval: ', pval)

