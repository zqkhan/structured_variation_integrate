import os.path

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.cluster import normalized_mutual_info_score
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re

from utils.data import synthetic_data_two
from utils.distributions import gaussian_pdf
from utils.coding_costs_ari import categorical_parameter, gaussian_parameters, iMDL_cost, iMDL_cluster_cost

import argparse

visualize = True

np.seterr(all='raise')
np.random.seed(0)

feature_type_dict = {
    'Valence': 'C',
    'Arousal': 'C',
    'mean_RSA': 'C',
    'mean_IBI': 'C',
    'mean_PEP': 'C',
    'mean_LVET': 'C',
    'mean_SV': 'C',
    'mean_CO': 'C',
    'Social.2': 'D',
    'Posture': 'D',
    'Activity': 'D',
}


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
    data = data.fillna(0)
    data = data.values
    return data


def initialize(data, num_init, num_clusters, use_cols='All', feature_type_dict={}):
    if use_cols == 'All':
        use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                    'Social.2', 'Posture', 'Activity']
    feature_types = np.array([feature_type_dict[c] for c in use_cols])
    cont_indices = np.where(feature_types == 'C')[0]
    dis_indices = np.where(feature_types == 'D')[0]
    soc_ind = np.where(np.array(args.use_cols) == 'Social.2')[0]
    pos_ind = np.where(np.array(args.use_cols) == 'Posture')[0]
    act_ind = np.where(np.array(args.use_cols) == 'Activity')[0]
    num_cont = len(cont_indices)
    init_MDL = np.zeros(shape=num_init)
    init_clusters = []
    K = num_clusters
    for m in range(num_init):
        print('Running Initialization Iteration: ' + str(m))
        mu_index = np.random.choice(len(data), K, replace=False)
        if num_cont > 0:
            mus = [copy.deepcopy(data[mu_index[k], cont_indices]) for k in range(K)]
            sigmas = [np.ones(shape=num_cont, ) for k in range(K)]
        else:
            mus = []
            sigmas = []
        p_soc = [[0.5, 1 - 0.5] for k in range(K)]
        p_pos = [[1 / 3, 1 / 3, 1 / 3] for k in range(K)]
        p_act = [[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5] for k in range(K)]
        c_ids = -1 * np.ones(len(data))
        for k in range(K):
            c_ids[mu_index[k]] = k
        clusters = {'mus': mus,
                    'sigmas': sigmas,
                    'p_soc': p_soc,
                    'p_pos': p_pos,
                    'p_act': p_act,
                    'c_ids': c_ids,
                    }
        temp_c_ids = copy.deepcopy(c_ids)
        for i in range(len(data)):
            # print(i)
            mdl = np.zeros(shape=(K))
            if i not in mu_index:
                for k in range(K):
                    temp_clusters = {'mus': copy.deepcopy(mus),
                                     'sigmas': copy.deepcopy(sigmas),
                                     'p_soc': copy.deepcopy(p_soc),
                                     'p_pos': copy.deepcopy(p_pos),
                                     'p_act': copy.deepcopy(p_act),
                                     'c_ids': copy.deepcopy(temp_c_ids),
                                     }
                    temp_clusters['c_ids'][i] = k
                    for j in cont_indices:
                        temp_clusters['mus'][k][j], temp_clusters['sigmas'][k][j] = gaussian_parameters(
                            data[np.where(temp_clusters['c_ids'] == k)[0], j])
                    if soc_ind.size > 0:
                        temp_clusters['p_soc'][k] = categorical_parameter(data[np.where(temp_clusters['c_ids'] == k)[0],
                        soc_ind.item()], [0, 1])
                    if pos_ind.size > 0:
                        temp_clusters['p_pos'][k] = categorical_parameter(data[np.where(temp_clusters['c_ids'] == k)[0],
                        pos_ind.item()], [0, 1, 2])
                    if act_ind.size > 0:
                        temp_clusters['p_act'][k] = categorical_parameter(data[np.where(temp_clusters['c_ids'] == k)[0],
                        act_ind.item()], [0, 1, 2, 3, 4])
                    mdl[k] = iMDL_cluster_cost(data, temp_clusters, k=k,
                                               cont_indices=cont_indices,
                                               soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)
                temp_c_ids[i] = np.argmin(mdl)

        clusters['c_ids'] = temp_c_ids
        for k in range(K):
            for j in cont_indices:
                clusters['mus'][k][j], clusters['sigmas'][k][j] = gaussian_parameters(
                    data[np.where(clusters['c_ids'] == k)[0], j])
            if soc_ind.size > 0:
                clusters['p_soc'][k] = categorical_parameter(data[np.where(clusters['c_ids'] == k)[0], soc_ind.item()],
                                                             [0, 1])
            if pos_ind.size > 0:
                clusters['p_pos'][k] = categorical_parameter(data[np.where(clusters['c_ids'] == k)[0], pos_ind.item()],
                                                             [0, 1, 2])
            if act_ind.size > 0:
                clusters['p_act'][k] = categorical_parameter(data[np.where(clusters['c_ids'] == k)[0], act_ind.item()],
                                                             [0, 1, 2, 3, 4])
        init_MDL[m] = iMDL_cost(data, clusters, K=K,
                                cont_indices=cont_indices,
                                soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)
        init_clusters.append(clusters)

    initial_clusters = init_clusters[np.argmin(init_MDL)]
    initial_MDL = np.min(init_MDL)
    print(initial_MDL)
    return initial_clusters


def optimize_integrate(data, current_clusters, num_iter, num_clusters, iter_thr=200, change_thr=2.0, use_cols='All',
                       feature_type_dict={}):
    N = len(data)
    K = num_clusters
    if use_cols == 'All':
        use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                    'Social.2', 'Posture', 'Activity']
    feature_types = np.array([feature_type_dict[c] for c in use_cols])
    cont_indices = np.where(feature_types == 'C')[0]
    dis_indices = np.where(feature_types == 'D')[0]
    soc_ind = np.where(np.array(args.use_cols) == 'Social.2')[0]
    pos_ind = np.where(np.array(args.use_cols) == 'Posture')[0]
    act_ind = np.where(np.array(args.use_cols) == 'Activity')[0]
    iter_costs = []
    for itr in range(num_iter):
        iter_start_assignments = copy.deepcopy(current_clusters['c_ids'])
        temp_c_ids = copy.deepcopy(current_clusters['c_ids'])
        mus = copy.deepcopy(current_clusters['mus'])
        sigmas = copy.deepcopy(current_clusters['sigmas'])
        p_soc = copy.deepcopy(current_clusters['p_soc'])
        p_pos = copy.deepcopy(current_clusters['p_pos'])
        p_act = copy.deepcopy(current_clusters['p_act'])
        temp_clusters = {'mus': copy.deepcopy(mus),
                         'sigmas': copy.deepcopy(sigmas),
                         'p_soc': copy.deepcopy(p_soc),
                         'p_pos': copy.deepcopy(p_pos),
                         'p_act': copy.deepcopy(p_act),
                         'c_ids': copy.deepcopy(temp_c_ids),
                         }
        for i in range(len(data)):
            iter_start_mdl = iMDL_cost(data, temp_clusters, K=K,
                                       cont_indices=cont_indices,
                                       soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)
            mdl = np.zeros(shape=(K))
            temp_mus = copy.deepcopy(temp_clusters['mus'])
            temp_sigmas = copy.deepcopy(temp_clusters['sigmas'])
            temp_p_soc = copy.deepcopy(temp_clusters['p_soc'])
            temp_p_pos = copy.deepcopy(temp_clusters['p_pos'])
            temp_p_act = copy.deepcopy(temp_clusters['p_act'])
            temp_temp_c_ids = copy.deepcopy(temp_clusters['c_ids'])
            current_k = int(temp_temp_c_ids[i])
            for k in range(K):
                temporary_clusters = {'mus': copy.deepcopy(temp_mus),
                                      'sigmas': copy.deepcopy(temp_sigmas),
                                      'p_soc': copy.deepcopy(temp_p_soc),
                                      'p_pos': copy.deepcopy(temp_p_pos),
                                      'p_act': copy.deepcopy(temp_p_act),
                                      'c_ids': copy.deepcopy(temp_temp_c_ids),
                                      }
                temporary_clusters['c_ids'][i] = k
                for j in cont_indices:
                    temporary_clusters['mus'][k][j], temporary_clusters['sigmas'][k][j] = gaussian_parameters(
                        data[np.where(temporary_clusters['c_ids'] == k)[0], j])
                if soc_ind.size > 0:
                    temporary_clusters['p_soc'][k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids'] == k)[0], soc_ind.item()],
                        [0, 1])
                if pos_ind.size > 0:
                    temporary_clusters['p_pos'][k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids'] == k)[0], pos_ind.item()],
                        [0, 1, 2])
                if act_ind.size > 0:
                    temporary_clusters['p_act'][k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids'] == k)[0], act_ind.item()],
                        [0, 1, 2, 3, 4])

                for j in cont_indices:
                    temporary_clusters['mus'][current_k][j], temporary_clusters['sigmas'][current_k][
                        j] = gaussian_parameters(
                        data[np.where(temporary_clusters['c_ids'] == current_k)[0], j])
                if soc_ind.size > 0:
                    temporary_clusters['p_soc'][current_k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids']
                                      == current_k)[0], soc_ind.item()],
                        [0, 1])
                if pos_ind.size > 0:
                    temporary_clusters['p_pos'][current_k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids']
                                      == current_k)[0], pos_ind.item()],
                        [0, 1, 2])
                if act_ind.size > 0:
                    temporary_clusters['p_act'][current_k] = categorical_parameter(
                        data[np.where(temporary_clusters['c_ids']
                                      == current_k)[0], act_ind.item()],
                        [0, 1, 2, 3, 4])
                mdl[k] = iMDL_cost(data, temporary_clusters, K=K,
                                   cont_indices=cont_indices,
                                   soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)

            min_k = np.argmin(mdl)
            current_clusters['c_ids'][i] = min_k
            temp_clusters['c_ids'][i] = min_k

            for j in cont_indices:
                temp_clusters['mus'][min_k][j], temp_clusters['sigmas'][min_k][j] = gaussian_parameters(
                    data[np.where(temp_clusters['c_ids'] == min_k)[0], j])

            if soc_ind.size > 0:
                temp_clusters['p_soc'][min_k] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == min_k)[0], soc_ind.item()],
                    [0, 1])
            if pos_ind.size > 0:
                temp_clusters['p_pos'][min_k] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == min_k)[0], pos_ind.item()],
                    [0, 1, 2])
            if act_ind.size > 0:
                temp_clusters['p_act'][min_k] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == min_k)[0], act_ind.item()],
                    [0, 1, 2, 3, 4])

            for j in cont_indices:
                temp_clusters['mus'][int(temp_c_ids[i])][j], temp_clusters['sigmas'][int(temp_c_ids[i])][
                    j] = gaussian_parameters(
                    data[np.where(temp_clusters['c_ids'] == int(temp_c_ids[i]))[0], j])

            if soc_ind.size > 0:
                temp_clusters['p_soc'][int(temp_c_ids[i])] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == int(temp_c_ids[i]))[0], soc_ind.item()], [0, 1])
            if pos_ind.size > 0:
                temp_clusters['p_pos'][int(temp_c_ids[i])] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == int(temp_c_ids[i]))[0], pos_ind.item()], [0, 1, 2])
            if act_ind.size > 0:
                temp_clusters['p_act'][int(temp_c_ids[i])] = categorical_parameter(
                    data[np.where(temp_clusters['c_ids'] == int(temp_c_ids[i]))[0], act_ind.item()], [0, 1, 2, 3, 4])

            if iter_start_mdl < iMDL_cost(data, temp_clusters, K=K,
                                          cont_indices=cont_indices,
                                          soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind):
                print(iMDL_cost(data, temp_clusters, K=K,
                                cont_indices=cont_indices,
                                soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind))
        if np.all(current_clusters['c_ids'] == iter_start_assignments):
            break
        else:
            assignments_changed = ((current_clusters['c_ids'] != iter_start_assignments).sum() / N) * 100
            for k in range(K):
                if len(np.where(current_clusters['c_ids'] == k)[0]) > 1:
                    for j in cont_indices:
                        current_clusters['mus'][k][j], current_clusters['sigmas'][k][j] = gaussian_parameters(
                            data[np.where(current_clusters['c_ids'] == k)[0], j])
                    if soc_ind.size > 0:
                        current_clusters['p_soc'][k] = categorical_parameter(
                            data[np.where(current_clusters['c_ids'] == k)[0], soc_ind.item()], [0, 1])
                    if pos_ind.size > 0:
                        current_clusters['p_pos'][k] = categorical_parameter(
                            data[np.where(current_clusters['c_ids'] == k)[0], pos_ind.item()], [0, 1, 2])
                    if act_ind.size > 0:
                        current_clusters['p_act'][k] = categorical_parameter(
                            data[np.where(current_clusters['c_ids'] == k)[0], act_ind.item()], [0, 1, 2, 3, 4])
            iter_costs.append(iMDL_cost(data, current_clusters, K=K,
                                        cont_indices=cont_indices,
                                        soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind))
            print("Iteration Number: " + str(itr) +
                  " Current MDL: " + str(iter_costs[-1]) +
                  " %ID Change: " + str(assignments_changed))
            if itr > iter_thr:
                if assignments_changed < change_thr:
                    break

    return current_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--participant', type=int, default=2)
    parser.add_argument('--range_num_clusters', default=range(1, 11))
    parser.add_argument('--num_init', type=int, default=10)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--directory', type=str, default='data/')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--change_thr', type=float, default=0.0)
    parser.add_argument('--iter_thr', type=int, default=1000)
    parser.add_argument('--use_cols', default='All', help="provide list of columns to use")
    args = parser.parse_args()
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    clustering_results = []
    mdl_costs = []
    if len(re.findall(r'\d+', args.results_dir)) > 0:
        raise ValueError("Please do not include numbers in results directory name")
    if args.use_cols == 'All':
        args.use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                         'Social.2', 'Posture', 'Activity']
    elif args.use_cols == 'Categorical':
        args.use_cols = ['Social.2', 'Posture', 'Activity']
        args.results_dir = 'results_cat_only/'
    elif args.use_cols == 'Continuous':
        args.use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO']
        args.results_dir = 'results_cont_only/'
    elif args.use_cols == 'Physio':
        args.use_cols = ['mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO']
        args.results_dir = 'results_phys_only/'
    elif args.use_cols == 'Affect':
        args.use_cols = ['Valence', 'Arousal']
        args.results_dir = 'results_val_aro/'
    else:
        raise ValueError("Please manually provide the list of features and update results directory accordingly")
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    feature_types = np.array([feature_type_dict[c] for c in args.use_cols])
    cont_indices = np.where(feature_types == 'C')[0]
    dis_indices = np.where(feature_types == 'D')[0]
    soc_ind = np.where(np.array(args.use_cols) == 'Social.2')[0]
    pos_ind = np.where(np.array(args.use_cols) == 'Posture')[0]
    act_ind = np.where(np.array(args.use_cols) == 'Activity')[0]
    for num_k in args.range_num_clusters:
        print('Running for K = ' + str(num_k))
        data = load_ari_data(participant=args.participant, data_directory=args.directory, use_cols=args.use_cols)
        initial_clusters = initialize(data=data, num_init=args.num_init, num_clusters=num_k, use_cols=args.use_cols,
                                      feature_type_dict=feature_type_dict)
        final_clusters = optimize_integrate(data=data, current_clusters=initial_clusters,
                                            num_iter=args.num_iter, num_clusters=num_k,
                                            iter_thr=args.iter_thr, change_thr=args.change_thr, use_cols=args.use_cols,
                                            feature_type_dict=feature_type_dict)
        clustering_results.append(final_clusters)
        print('...................................................................................')
        mdl = iMDL_cost(data, final_clusters, K=num_k,
                        cont_indices=cont_indices,
                        soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)
        mdl_costs.append(mdl)
        print('Final MDL Cost: ' + str(mdl))
        print('...................................................................................')

    plt.plot(args.range_num_clusters, mdl_costs)
    plt.savefig(args.results_dir + 'mdl_costs_k_' + str(args.participant) + '.PNG')
    plt.close()

    best_k = np.argmin(mdl_costs)
    save_file = open(args.results_dir + str(args.participant) + '_mdl_clusters.pk', "wb")
    pickle.dump(clustering_results[best_k], save_file)
    save_file.close()
