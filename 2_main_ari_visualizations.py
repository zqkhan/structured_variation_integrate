import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import glob
import re
import pandas as pd
from utils.coding_costs_ari import gaussian_parameters, categorical_parameter

show = False


def load_ari_data(participant=2, data_directory='data/', use_cols='All'):
    if use_cols == 'All':
        use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                    'Social.2', 'Posture', 'Activity']
    filename = data_directory + str(participant) + '_compiled.csv'
    data = pd.read_csv(filename)
    data['Social.2'] = data['Social.2'] - 1
    data['Posture'] = data['Posture'] - 1
    data['Activity'] = data['Activity'] - 1
    data = data[use_cols]
    data = data.values

    return data


num_cont = 8
all_features = ['Valence', 'Arousal', 'RSA', 'IBI', 'PEP', 'LVET', 'SV', 'CO', 'Social', 'Posture', 'Activity']
cat_features = [['not alone', 'alone'], ['sitting', 'standing', 'reclining'], ['non-work', 'work', 'leisure', 'eating',
                                                                               'computer']]
RESULTS_DIR = 'results/'  #choose appropriate results directory that you want to visualize
VISUALIZATIONS_DIR = 'visualizations/'  #name appropriately for visualizations to go to
if not os.path.isdir(VISUALIZATIONS_DIR):
    os.mkdir(VISUALIZATIONS_DIR)
soc_color_list = ['r', 'g']
pos_color_list = ['c', 'm', 'y']
act_color_list = ['b', 'orange', 'lime', 'mistyrose', 'black']
for results_file in glob.glob(RESULTS_DIR + '*.pk'):
    p = int(re.findall(r'\d+', results_file)[0])
    print('Visualizing Participant: ' + str(p))
    results = pickle.load(open(results_file, 'rb'))
    data = load_ari_data(participant=p, )

    K = len(np.unique(results['c_ids']))

    summary = {
        'mus': [np.zeros(shape=(8,)) for k in range(K + 1)],
        'sigmas': [np.zeros(shape=(8,)) for k in range(K + 1)],
        'p_soc': [[0.5, 1 - 0.5] for k in range(K + 1)],
        'p_pos': [[1 / 3, 1 / 3, 1 / 3] for k in range(K + 1)],
        'p_act': [[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5] for k in range(K + 1)],
        'cluster_weights': np.array([100 * ((results['c_ids'] == k).sum() / len(data)) for k in range(K)])
    }
    sort_k = summary['cluster_weights'].argsort()[::-1]
    for (oldk, k) in enumerate(sort_k):
        for j in range(num_cont):
            summary['mus'][oldk][j], summary['sigmas'][oldk][j] = gaussian_parameters(
                data[np.where(results['c_ids'] == k)[0], j])
        summary['p_soc'][oldk] = categorical_parameter(data[np.where(results['c_ids'] == k)[0], 8],
                                                       attributes_values=[0, 1])
        summary['p_pos'][oldk] = categorical_parameter(data[np.where(results['c_ids'] == k)[0], 9],
                                                       attributes_values=[0, 1, 2])
        summary['p_act'][oldk] = categorical_parameter(data[np.where(results['c_ids'] == k)[0], 10],
                                                       attributes_values=[0, 1, 2, 3, 4])

    for j in range(num_cont):
        summary['mus'][-1][j], summary['sigmas'][-1][j] = gaussian_parameters(
            data[:, j])
    summary['p_soc'][-1] = categorical_parameter(data[:, 8],
                                                 attributes_values=[0, 1])
    summary['p_pos'][-1] = categorical_parameter(data[:, 9],
                                                 attributes_values=[0, 1, 2])
    summary['p_act'][-1] = categorical_parameter(data[:, 10],
                                                 attributes_values=[0, 1, 2, 3, 4])
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True)
    axes = axes.reshape(3 * 4)

    for (i, ax) in enumerate(axes):
        xlabels = np.arange(K + 1)
        if i < len(axes) - 1:
            ax.axvline(x=K - 0.5, color='black', linestyle='dashed')
        if i < num_cont:
            ax.set_xlabel(all_features[i])
            mean_values = [m[i] for m in summary['mus']]
            std_values = [s[i] for s in summary['sigmas']]
            ax.bar(xlabels, mean_values, yerr=std_values)
        elif i == 8:
            ax.set_xlabel(all_features[i])
            probs = [np.array([pr[0] for pr in summary['p_soc']]), np.array([pr[1] for pr in summary['p_soc']])]
            ax.bar(xlabels, probs[0], width=0.25, color=soc_color_list[0])
            ax.bar(xlabels, probs[1], width=0.25, bottom=probs[0], color=soc_color_list[1])
        elif i == 9:
            ax.set_xlabel(all_features[i])
            probs = [np.array([pr[0] for pr in summary['p_pos']]), np.array([pr[1] for pr in summary['p_pos']]),
                     np.array([pr[2] for pr in summary['p_pos']])]
            ax.bar(xlabels, probs[0], width=0.25, color=pos_color_list[0])
            ax.bar(xlabels, probs[1], width=0.25, bottom=probs[0], color=pos_color_list[1])
            ax.bar(xlabels, probs[2], width=0.25, bottom=probs[1] + probs[0], color=pos_color_list[2])
        elif i == 10:
            ax.set_xlabel(all_features[i])
            probs = [np.array([pr[0] for pr in summary['p_act']]), np.array([pr[1] for pr in summary['p_act']]),
                     np.array([pr[2] for pr in summary['p_act']]), np.array([pr[3] for pr in summary['p_act']]),
                     np.array([pr[4] for pr in summary['p_act']])]
            ax.bar(xlabels, probs[0], width=0.25, color=act_color_list[0])
            ax.bar(xlabels, probs[1], width=0.25, bottom=probs[0], color=act_color_list[1])
            ax.bar(xlabels, probs[2], width=0.25, bottom=probs[1] + probs[0], color=act_color_list[2])
            ax.bar(xlabels, probs[3], width=0.25, bottom=probs[2] + probs[1] + probs[0], color=act_color_list[3])
            ax.bar(xlabels, probs[4], width=0.25, bottom=probs[3] + probs[2] + probs[1] + probs[0],
                   color=act_color_list[4])
        else:
            social_red_patch = mpatches.Patch(color=soc_color_list[0], label='Not alone')
            social_green_patch = mpatches.Patch(color=soc_color_list[1], label='Alone')

            pos_red_patch = mpatches.Patch(color=pos_color_list[0], label='Sitting')
            pos_green_patch = mpatches.Patch(color=pos_color_list[1], label='Standing')
            pos_blue_patch = mpatches.Patch(color=pos_color_list[2], label='Reclining')

            act_red_patch = mpatches.Patch(color=act_color_list[0], label='Non-work')
            act_green_patch = mpatches.Patch(color=act_color_list[1], label='Work')
            act_blue_patch = mpatches.Patch(color=act_color_list[2], label='Leisure')
            act_yellow_patch = mpatches.Patch(color=act_color_list[3], label='Eating')
            act_black_patch = mpatches.Patch(color=act_color_list[4], label='Computer')

            legend1 = ax.legend(handles=[social_red_patch, social_green_patch], loc='upper left', prop={'size': 10})
            legend2 = ax.legend(handles=[pos_red_patch, pos_green_patch, pos_blue_patch], loc='upper right',
                                prop={'size': 10})
            legend3 = ax.legend(
                handles=[act_red_patch, act_green_patch, act_blue_patch, act_yellow_patch, act_black_patch],
                loc='lower center', prop={'size': 10})

            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(
            np.hstack([np.round(np.sort(np.array(summary['cluster_weights']))[::-1], 2), np.array([100, ])]))
    plt.suptitle('Participant: ' + str(p) + ' Total Events: ' + str(len(data)))
    plt.tight_layout()
    plt.subplots_adjust(top=0.945, bottom=0.072, left=0.038, right=0.990, hspace=0.191, wspace=0.213)
    fig.set_size_inches(12, 8)
    if show:
        plt.show()
    else:
        plt.savefig(VISUALIZATIONS_DIR + str(p) + '_cluster_summary.PNG')
        plt.close()
r = 3
