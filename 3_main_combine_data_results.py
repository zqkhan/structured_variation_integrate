import numpy as np
import glob
import re
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_ari_data(participant=2, data_directory='data/', standardize=True, use_cols='All'):
    if use_cols == 'All':
        use_cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
                    'Social.2', 'Posture', 'Activity']
    filename = data_directory + str(participant) + '_compiled.csv'
    data = pd.read_csv(filename)
    if standardize:
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


STANDARDIZE = True  # set this to false if you want original input data
RESULTS_DIR = 'results/'  #set this to where the cluster pickles are located

num_cont = 8
all_features = ['class', 'Valence', 'Arousal', 'RSA', 'IBI', 'PEP', 'LVET', 'SV', 'CO', 'Social', 'Posture', 'Activity']
cat_features = [['not alone', 'alone'], ['sitting', 'standing', 'reclining'], ['non-work', 'work', 'leisure', 'eating',
                                                                               'computer']]

p_list = [int(re.findall(r'\d+', results_file)[0])
          for results_file in glob.glob(RESULTS_DIR + '/*_mdl_clusters.pk')]
file_list = np.array(glob.glob(RESULTS_DIR + '/*_mdl_clusters.pk'))[np.argsort(p_list)]

for results_file in file_list:
    p = int(re.findall(r'\d+', results_file)[0])
    print('Combining data for Participant: ' + str(p))
    results = pickle.load(open(results_file, 'rb'))
    data = load_ari_data(participant=p, standardize=STANDARDIZE)
    data_results = np.hstack((np.expand_dims(results['c_ids'], -1), data))
    data_results_pd = pd.DataFrame(data_results, columns=all_features)
    if STANDARDIZE:
        data_results_pd.to_csv(RESULTS_DIR + str(p) + '_data_results.csv', index=False)
    else:
        data_results_pd.to_csv(RESULTS_DIR + str(p) + '_data_results_original.csv', index=False)
    r = 3
