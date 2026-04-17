import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

def normalized_mutual_info_cont(feature, labels,
                                discrete_features=np.array([False, False, False, False, False, False, False, False,
                                                            True, True, True]), random_state=0, no_co=False):
    if no_co:
        discrete_features = np.array([False, False, False, False, False, False, False,
                                                            True, True, True])
    else:
        discrete_features = discrete_features
    labels = labels.astype(int)
    mi = mutual_info_classif(feature, labels, discrete_features=discrete_features, random_state=random_state)
    nmi = (1 - np.exp(-2 * mi)) ** (1/2)

    return nmi

def calculate_q95(shuffled_nmi):

    shuffled_nmi_means = shuffled_nmi.mean(axis=0)
    shuffled_nmi_stds = shuffled_nmi.std(axis=0)
    q95 = np.zeros(shape=(len(shuffled_nmi_means),))
    for f in range(len(shuffled_nmi_means)):
        q95[f] = stats.norm.ppf(0.95, shuffled_nmi_means[f], shuffled_nmi_stds[f])

    return q95

def calculate_p_sig(nmi, shuffled_nmi):

    p_sig = np.zeros(shape=(len(nmi),))
    for f in range(len(nmi)):
        p_sig[f] = 1 - (nmi[f] > shuffled_nmi[:, f]).mean()

    return p_sig

def calculate_p_sig_jmi(true_rankings, shuffled_rankings):
    true_rankings = np.array(true_rankings)
    p_sig = np.zeros(shape=(len(true_rankings),))
    for (i, f) in enumerate(true_rankings):
        p_sig[i] = (np.where(shuffled_rankings == f)[1] <= i).mean()

    return p_sig

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]
