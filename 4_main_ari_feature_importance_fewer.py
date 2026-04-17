import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import mifs
import pandas as pd
import os
from utils.helpers import normalized_mutual_info_cont, calculate_p_sig, zero_to_nan, calculate_p_sig_jmi

show = False
p_thr1 = 0.01
p_thr2 = 0.05
num_permute_nmi = 1000
num_permute_jmi = 1000

def load_ari_data(participant=2, data_directory='data/', use_cols='All'):
    if use_cols == 'All':
        use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                    'Social.2', 'Posture', 'Activity']
    filename = data_directory + str(participant) + '_compiled.csv'
    data = pd.read_csv(filename)
    data['Social.2'] = data['Social.2'] - 1
    data['Posture'] = data['Posture'] - 1
    data['Activity'] = data['Activity'] - 1
    data[['Valence', 'Arousal']] = (data[['Valence', 'Arousal']] +
                                    np.random.multivariate_normal(mean=np.array([0, 0, ]), cov=0.1 * np.eye(2),
                                                                   size=len(data)))
    data[['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO']] = (
        StandardScaler().fit_transform(data[['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET',
                                          'mean_SV', 'mean_CO']]))
    data = data[use_cols]
    data = data.values
    return data

USE_COLS = 'All'
num_cont = 8
all_features = ['Valence', 'Arousal', 'RSA', 'IBI', 'PEP', 'LVET', 'SV', 'CO', 'Social', 'Posture', 'Activity']
discrete_features=np.array([False, False, False, False, False, False, False, False,
                                                            True, True, True])
cat_features = [['not alone', 'alone'], ['sitting', 'standing', 'reclining'], ['non-work', 'work', 'leisure', 'eating',
                                                                               'computer']]
RESULTS_DIR = 'results/'
VISUALIZATIONS_DIR = 'visualizations/'
soc_color_list = ['r', 'g']
pos_color_list = ['c', 'm', 'y']
act_color_list = ['b', 'orange', 'lime', 'mistyrose', 'black']

P_LIST = [2,]

participant_feature_means = {p: [] for p in P_LIST}
participant_feature_stds = {p: [] for p in P_LIST}
participant_feature_psigs = {p: [] for p in P_LIST}
participant_feature_jmi_psigs = {p: [] for p in P_LIST}
participant_feature_ranks_jmi = {p: [] for p in P_LIST}
participant_feature_importance_jmi = {p: [] for p in P_LIST}
participant_feature_ranks_jmib = {p: [] for p in P_LIST}
participant_feature_importance_jmib = {p: [] for p in P_LIST}

k_1_count = 0

for p in P_LIST:
    results_file = RESULTS_DIR + '/' + str(p) + '_mdl_clusters.pk'
    if os.path.isfile(results_file):
        print('Calculating Feature Importance for Participant: ' + str(p))
        results = pickle.load(open(results_file, 'rb'))
        K = len(np.unique(results['c_ids']))
        if K > 1:
            data = load_ari_data(participant=p, use_cols=USE_COLS)
            knn_k_alternate = int(np.min(np.unique(results['c_ids'], return_counts=True)[1]) - 1)
            knn_k = 5
            try:
                feature_selector = mifs.MutualInformationFeatureSelector(method='JMI', k=knn_k, n_features=11,
                                                                         categorical=True, verbose=0)
                feature_selector.fit(data, results['c_ids'].astype(int), discrete_features=discrete_features)
                participant_feature_importance_jmi[p] = np.cumsum(feature_selector.mi_)
                participant_feature_ranks_jmi[p] = feature_selector.ranking_

                K = len(np.unique(results['c_ids']))
                shuffled_jmi_ranking = np.empty(shape=(num_permute_jmi, len(all_features)))
                for s in tqdm(range(len(shuffled_jmi_ranking))):
                    shuffled_cids = np.random.permutation(results['c_ids'])
                    shuffled_jmi_ranking[s, :] = feature_selector.fit_compare_fixed_order(data, shuffled_cids.astype(int),
                                                     discrete_features=discrete_features)

                participant_feature_jmi_psigs[p] = calculate_p_sig_jmi(feature_selector.ranking_, shuffled_jmi_ranking)

                feature_selector = mifs.MutualInformationFeatureSelector(method='JMI', k=knn_k, n_features=11,
                                                                         categorical=True, verbose=0)
                feature_selector.fit_backward(data, results['c_ids'].astype(int), discrete_features=discrete_features)
                participant_feature_importance_jmib[p] = feature_selector.mi_
                participant_feature_ranks_jmib[p] = feature_selector.ranking_

            except:
                print('using alternate k')
                feature_selector = mifs.MutualInformationFeatureSelector(method='JMI', k=knn_k_alternate, n_features=11,
                                                                         categorical=True, verbose=0)
                feature_selector.fit(data, results['c_ids'].astype(int), discrete_features=discrete_features)
                participant_feature_importance_jmi[p] = np.cumsum(feature_selector.mi_)
                participant_feature_ranks_jmi[p] = feature_selector.ranking_
                K = len(np.unique(results['c_ids']))
                shuffled_jmi_ranking = np.empty(shape=(num_permute_jmi, len(all_features)))
                for s in tqdm(range(len(shuffled_jmi_ranking))):
                    shuffled_cids = np.random.permutation(results['c_ids'])
                    shuffled_jmi_ranking[s, :] = feature_selector.fit_compare_fixed_order(data, shuffled_cids.astype(int),
                                                     discrete_features=discrete_features)

                participant_feature_jmi_psigs[p] = calculate_p_sig_jmi(feature_selector.ranking_, shuffled_jmi_ranking)

                feature_selector = mifs.MutualInformationFeatureSelector(method='JMI', k=knn_k_alternate, n_features=11,
                                                                         categorical=True, verbose=0)
                feature_selector.fit_backward(data, results['c_ids'].astype(int), discrete_features=discrete_features)
                participant_feature_importance_jmib[p] = feature_selector.mi_
                participant_feature_ranks_jmib[p] = feature_selector.ranking_
            feature_importance = []
            for r in range(100):
                feature_importance.append(normalized_mutual_info_cont(data, results['c_ids'], random_state=r))
            feature_importance_mean = np.array(feature_importance).mean(axis=0)
            feature_importance_std = np.array(feature_importance).std(axis=0)

            K = len(np.unique(results['c_ids']))
            shuffled_nmi = np.empty(shape=(num_permute_nmi, len(all_features)))
            for s in tqdm(range(len(shuffled_nmi))):
                shuffled_cids = np.random.permutation(results['c_ids'])
                shuffled_nmi[s, :] = normalized_mutual_info_cont(data, shuffled_cids)

            participant_feature_psigs[p] = calculate_p_sig(feature_importance_mean, shuffled_nmi)

            # q95 = calculate_q95(shuffled_nmi)
            participant_feature_means[p] = feature_importance_mean
            participant_feature_stds[p] = feature_importance_std
            K = len(np.unique(results['c_ids']))
        else:
            print("No Clustering")
            k_1_count += 1

for keys in participant_feature_ranks_jmi.keys():
    pickle.dump(participant_feature_ranks_jmi[keys], file=open(RESULTS_DIR + str(keys) +
                                                               '_pickled_scores_' + 'jmi_rank.pk', 'wb'))
    pickle.dump(participant_feature_ranks_jmib[keys], file=open(RESULTS_DIR+ str(keys) +
                                                                '_pickled_scores_' + 'jmib_rank.pk', 'wb'))
    pickle.dump(participant_feature_importance_jmi[keys], file=open(RESULTS_DIR + str(keys) +
                                                                    '_pickled_scores_' + 'jmi.pk', 'wb'))
    pickle.dump(participant_feature_importance_jmib[keys], file=open(RESULTS_DIR + str(keys) +
                                                                     '_pickled_scores_' + 'jmi.pk', 'wb'))
    pickle.dump(participant_feature_jmi_psigs[keys], file=open(RESULTS_DIR + str(keys) +
                                                               '_pickled_scores_' + 'jmi_psigs.pk', 'wb'))
    pickle.dump(participant_feature_means[keys], file=open(RESULTS_DIR + str(keys) +
                                                           '_pickled_scores_' + 'nmi_means.pk', 'wb'))
    pickle.dump(participant_feature_stds[keys], file=open(RESULTS_DIR + str(keys) +
                                                          '_pickled_scores_' + 'nmi_stds.pk', 'wb'))
    pickle.dump(participant_feature_psigs[keys], file=open(RESULTS_DIR + str(keys) +
                                                           '_pickled_scores_' + 'nmi_psigs.pk', 'wb'))


xlabels = np.arange(len(all_features))
i_ax = 0
i_bx = 0
for p in P_LIST:
    results_file = RESULTS_DIR + '/' + str(p) + '_mdl_clusters.pk'
    if os.path.isfile(results_file):
        print('Visualizing Participant: ' + str(p))
        results = pickle.load(open(results_file, 'rb'))
        K = len(np.unique(results['c_ids']))
        if K > 1:
            fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)
            axes = axes.reshape(1 * 3)

            ax = axes[0]
            top_k = participant_feature_means[p].argsort()[::-1]
            p_sig = participant_feature_psigs[p][top_k]
            ax.plot(participant_feature_means[p][top_k], 'o')
            ax.plot(zero_to_nan(participant_feature_means[p][top_k] * (p_sig <= p_thr1)), 'o', color='red')
            ax.plot(zero_to_nan(participant_feature_means[p][top_k] * ((p_sig > p_thr1)
                                                                       & (p_sig <= p_thr2))),
                    'o', color='green')
            ax.set_xticks(xlabels)
            ax.set_xticklabels(np.array(all_features)[top_k], rotation=45)
            ax.set_title('NMI, Green p<0.05, Red p<0.01')

            ax = axes[1]
            p_sig = participant_feature_jmi_psigs[p]
            ax.plot(participant_feature_importance_jmi[p], 'o')
            # ax.plot(zero_to_nan(participant_feature_importance_jmi[p] * (p_sig <= p_thr1)), 'o', color='red')
            # ax.plot(zero_to_nan(participant_feature_importance_jmi[p] * ((p_sig > p_thr1)
            #                                                            & (p_sig <= p_thr2))), 'o', color='green')

            top_k = participant_feature_ranks_jmi[p]
            ax.set_xticks(xlabels)
            ax.set_xticklabels(np.array(all_features)[top_k], rotation=45)
            ax.set_title('Joint Mutual Information (JMI)')

            ax = axes[2]
            ax.plot(participant_feature_jmi_psigs[p], 'x')
            top_k = participant_feature_ranks_jmi[p]
            ax.set_xticks(xlabels)
            ax.set_xticklabels(np.array(all_features)[top_k], rotation=45)
            ax.set_title('JMI Random Shuffling %age Ranks')

            fig.suptitle('P = ' + str(p) + ', K = ' + str(K) + ', N = ' + str(len(results['c_ids'])))
            # fig.tight_layout()
            fig.set_size_inches(12, 8)
            fig.subplots_adjust(top=0.919, bottom=0.090, left=0.039, right=0.990, hspace=0.322, wspace=0.102)
            fig.savefig(VISUALIZATIONS_DIR + str(p) + '_feature_importance_nmi_jmi.png')



