# Author: Philip Deming 

import sys
sys.path.append(".")

import os, glob
import numpy as np
import pandas as pd
import pickle
from utils.coding_costs_ari import iMDL_cost
from utils.coding_costs_ari import iMDL_cluster_cost
from utils.coding_costs_ari import gaussian_parameters
from utils.coding_costs_ari import categorical_parameter
from sklearn.preprocessing import StandardScaler
from itertools import combinations

#set variables
cols = ['Valence', 'Arousal', 'mean_RSA', 'mean_IBI', 'mean_PEP', 'mean_LVET', 'mean_SV', 'mean_CO',
        'Social.2', 'Posture', 'Activity']
cont_ind = range(8)
soc_ind = np.squeeze(np.where(np.array(cols)=='Social.2'))
pos_ind = np.squeeze(np.where(np.array(cols)=='Posture'))
act_ind = np.squeeze(np.squeeze(np.where(np.array(cols)=='Activity')))

#set paths
root_dir = 'project' #change to name of project root directory
data_dir = os.path.join(root_dir, f'data')
results_dir = os.path.join(root_dir, f'results')

#set files
fs = glob.glob(os.path.join(data_dir, '*compiled.csv'))
fs = sorted(fs)
subjects = [int(f.split('/')[-1].split('_')[0]) for f in fs]
subjects = sorted(subjects)

def load_orig_data(subj, data_directory, cols):
    filename = os.path.join(data_directory, f'{subj}_compiled.csv')
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
    data = data[cols]
    data = data.fillna(0)
    data = data.values
    return data

def load_clust_data(subj, data_directory):
    filename = os.path.join(data_directory, f'{subj}_mdl_clusters.pk')
    data = pickle.load(open(filename,'rb'))
    return data

def inter_sub_pairs(n1, n2):
    list=[]
    for i in range(n1):
        for j in range(n2):
            if (j,i) not in list:
                list.append((i,j))
    return list

def prepare_data_for_clust_pair(data, res_sum, clust, pair, U=False):
    """
    U determines whether the data are prepared to calculate iMDL jointly/separately for the cluster pair
    True=joint; False=separate
    """
    
    #select rows in original data
    pair_data = [d[np.where(clust[i]==pair[i])] for i, d in enumerate(data)]
    pair_data = np.concatenate((np.array(pair_data[0]), np.array(pair_data[1])))
    
    #prepare dictionary
    if U:
        pair_clust = np.zeros(len(pair_data))
        mus=np.empty(np.max(cont_ind)+1)
        sigmas=np.empty(np.max(cont_ind)+1)
        for j in cont_ind:
            mus[j], sigmas[j] = gaussian_parameters(pair_data[:,j])
        pair_dict = {
            'mus' : [mus],
            'sigmas' : [sigmas],
            'p_soc' : [categorical_parameter(pair_data[:,soc_ind.item()], [0,1])],
            'p_pos' : [categorical_parameter(pair_data[:,pos_ind.item()], [0,1,2])],
            'p_act' : [categorical_parameter(pair_data[:,act_ind.item()], [0,1,2,3,4])],
            'c_ids' : pair_clust
        }
    else:
        clust_len = [len(clust[i][np.where(clust[i]==pair[i])]) for i in range(2)]
        pair_clust = np.concatenate([np.zeros(clust_len[0]), np.ones(clust_len[1])])
        pair_dict = {
            'mus' : [res_sum[i]['mus'][pair[i]] for i in range(2)],
            'sigmas' : [res_sum[i]['sigmas'][pair[i]] for i in range(2)],
            'p_soc' : [res_sum[i]['p_soc'][pair[i]] for i in range(2)],
            'p_pos' : [res_sum[i]['p_pos'][pair[i]] for i in range(2)],
            'p_act' : [res_sum[i]['p_act'][pair[i]] for i in range(2)],
            'c_ids' : pair_clust
        }
    
    return pair_data, pair_dict

def scale_distance(data_len, imdl_joint, imdl_sep):
    distance = imdl_joint - imdl_sep
    scaled_distance = distance / data_len
    return scaled_distance

def scaled_distance_threshold(n_clust_A, n_clust_B):
    PC_merged = (23 / 2) * np.log(n_clust_A + n_clust_B)
    PC_separate = (23 / 2) * np.log(n_clust_A) + (23 / 2) * np.log(n_clust_B)
    PC_saved = PC_separate - PC_merged

    return PC_saved / (n_clust_A + n_clust_B)

def calc_imdl_cluster_pair_distance(sub1, sub2, data_directory, results_directory):
    data = [load_orig_data(sub, data_directory, cols) for sub in [sub1,sub2]]
    res_sum = [load_clust_data(sub, results_directory) for sub in [sub1,sub2]]
    #clust = [np.array(i['c_ids']).astype(int) for i in res_sum]
    clust = [i['c_ids'] for i in res_sum]

    #find number of clusters per subject
    n_clust = [int(np.max(i)+1) for i in clust]

    #make pairs of clusters
    if sub1==sub2:
        pairs = combinations(range(n_clust[0]),2)
    elif sub1!=sub2:
        pairs = inter_sub_pairs(n_clust[0],n_clust[1])

    #loop through cluster pairs
    imdl_sep_all=[]
    imdl_joint_all=[]
    distance_all=[]
    scaled_all = []
    scaled_thr_all = []
    for pair in pairs:

        ## separate iMDL
        #prepare data for current cluster pair
        pair_data, pair_dict = prepare_data_for_clust_pair(data=data, res_sum=res_sum, clust=clust, pair=pair)
        scaled_thr = scaled_distance_threshold((pair_dict['c_ids'] == 0).sum(), (pair_dict['c_ids'] == 1).sum())
        imdl_sep = iMDL_cost(data=pair_data, clusters=pair_dict, K=2, cont_indices=cont_ind, soc_ind=soc_ind, pos_ind=pos_ind, act_ind=act_ind)
        imdl_sep_all.append(imdl_sep)

        ## joint iMDL
        #prepare data for current cluster pair
        pair_data, pair_dict = prepare_data_for_clust_pair(data=data, res_sum=res_sum, clust=clust, pair=pair, U=True)
        imdl_joint = iMDL_cost(data=pair_data, clusters=pair_dict, K=1, cont_indices=cont_ind, soc_ind=soc_ind, pos_ind=pos_ind, act_ind=act_ind)
        imdl_joint_all.append(imdl_joint)

        ## scaled distance between joint iMDL and separate iMDL
        scaled = scale_distance(data_len=len(pair_data), imdl_joint=imdl_joint, imdl_sep=imdl_sep)
        scaled_all.append(scaled)
        scaled_thr_all.append(scaled_thr)
        
    ## output
    out = [np.repeat(sub1, len(imdl_joint_all)),np.repeat(sub2, len(imdl_joint_all)),
           np.array(pairs)[:,0],np.array(pairs)[:,1],
           imdl_joint_all,imdl_sep_all,scaled_all,scaled_thr_all]
    columns = ['sub1','sub2','cluster1','cluster2','imdl_joint','imdl_sep', 'scaled_distance', 'scaled_thr']

    return out, columns

## compute iMDL for all intersubject cluster pairs
mat = []
for i, sub1 in enumerate(subjects):
    for sub2 in subjects[i:]:            
        if sub1!=sub2:
            data, columns = calc_imdl_cluster_pair_distance(sub1, sub2, data_dir, results_dir)
            mat.append(np.array(data).T)
df = pd.DataFrame(np.concatenate(mat), columns=columns)

#save dataframe
df.to_csv(os.path.join(results_dir, 'scaled_distance.csv'), index=False)
