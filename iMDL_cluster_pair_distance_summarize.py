# Author: Philip Deming

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle

#set paths
root_dir = 'project' #change to name of project root directory
results_dir = os.path.join(root_dir, f'results')

def summarize_scaled_distance(df):
    
    #mean iMDL per subject (across clusters and cluster pairs)
    m = pd.concat([df.groupby(i).scaled_distance.mean() for i in ['sub1','sub2']], axis=1).mean(axis=1)
    
    #proportion of negative scaled distance values
    df['scaled_distance_neg'] = np.where(df['scaled_distance']<0, 1, 0)
    total_pairwise = pd.concat([df.groupby(i).scaled_distance_neg.count() for i in ['sub1','sub2']], axis=1).sum(axis=1)
    n_neg = pd.concat([df.groupby(i).scaled_distance_neg.sum() for i in ['sub1','sub2']], axis=1).sum(axis=1)
    p_neg = np.divide(n_neg, total_pairwise)
    
    return m, p_neg

def convert_to_adjacency_mat(df, val):
    
    #get list of all clusters
    sub1_cluster1 = np.array(df['sub1'].astype(int).astype(str) + "_" + df['cluster1'].astype(int).astype(str))
    sub2_cluster2 = np.array(df['sub2'].astype(int).astype(str) + "_" + df['cluster2'].astype(int).astype(str))
    sub_cluster_labels = np.unique(np.concatenate([sub1_cluster1,sub2_cluster2]))
    sub_cluster_labels = np.array(sorted(sub_cluster_labels, key=int))
    
    #assign numeric labels to each cluster
    sub1_cluster1_num = [np.squeeze(np.where(sub_cluster_labels==i)).item() for i in sub1_cluster1]
    sub2_cluster2_num = [np.squeeze(np.where(sub_cluster_labels==i)).item() for i in sub2_cluster2]
    
    #format output array
    df['sub1_cluster1_num'] = sub1_cluster1_num
    df['sub2_cluster2_num'] = sub2_cluster2_num
    out = df.pivot(index='sub1_cluster1_num', columns='sub2_cluster2_num', values=f'{val}').values
    
    #insert nan into missing rows at end of adjacency matrix
    missing_sub1_cluster1_clusters = len(sub_cluster_labels) - np.max(sub1_cluster1_num) - 1
    for i in range(missing_sub1_cluster1_clusters):
        out = np.insert(out.astype(float), len(out), np.full(out.shape[1], np.nan), axis=0)
        
    #insert nan into missing columns at beginning of adjacency matrix
    missing_sub2_cluster2_clusters = np.min(sub2_cluster2_num)
    for i in range(missing_sub2_cluster2_clusters):
        out = np.insert(out.astype(float), 0, np.full(out.shape[0], np.nan), axis=1)
        
    return out, sub_cluster_labels

def calc_cluster_prop(df, val, cut_point, num_clusters):
    print(f'Summarizing proportion of clusters based on variable: {val}')
    
    if val=='different':
        # create different variable, find cluster pairs where scaled distance > threshold
        df['different'] = np.where(df['scaled_distance'] > df['scaled_thr'], 1, 0)
        
    # prepare data
    mat, _ = convert_to_adjacency_mat(df, val)
    
    if val=='scaled_distance':
        # binarize scaled distance values (negative = 1, all else = 0)
        all_edges = np.where(mat<0, 1, 0)
    elif val=='different':
        # set all different edges to 1, others to 0 (just to remove nans)
        all_edges = np.where(mat==1, 1, 0)
    
    for cohort in range(2):
        
        if cohort==0:
            group_idx = np.arange(0,cut_point)
        elif cohort==1:
            group_idx = np.arange(cut_point,num_clusters)
                
        # isolate current cohort
        c_edges = all_edges[group_idx][:,group_idx]
        
        # count edges
        c_clusters = np.sum(c_edges, axis=0) + np.sum(c_edges, axis=1)
        c_clusters_per = np.round(c_clusters / len(c_clusters) * 100, 1)
        
        print(f'Cohort {cohort+1}: M:', np.round(np.mean(c_clusters),1), 
              'SD:', np.round(np.std(c_clusters),1), 
              'Min:', np.round(np.min(c_clusters),1),
              '  Max:', np.round(np.max(c_clusters),1))
        print(f'Cohort {cohort+1} (%): M:', np.round(np.mean(c_clusters_per),1), 
              'SD: ', np.round(np.std(c_clusters_per),1), 
              'Min:', np.round(np.min(c_clusters_per),1),
              'Max:', np.round(np.max(c_clusters_per),1))

## load data
df = pd.read_csv(os.path.join(results_dir, 'scaled_distance.csv'))

#find start of cohort 2 in df
cohort2_start = df[df['sub1']==201].sub1_cluster1_num.values[0]

## Summarize cluster pairs that are similar
calc_cluster_prop(df, val='scaled_distance', cut_point=cohort2_start, num_clusters=313)

## Summarize cluster pairs that are different
calc_cluster_prop(df, val='different', cut_point=cohort2_start, num_clusters=313)

## Community Detection

#prepare data
mat, sub_cluster_labels = convert_to_adjacency_mat(df, 'scaled_distance')

# set all negative iMDL edges to 1, others to 0
all_edges = np.where(mat<0, 1, 0)

# Cohort 1

# isolate Cohort 1
cohort1_edges = all_edges[:cohort2_start,:cohort2_start]

# Louvain Community Detection
G = nx.from_numpy_array(cohort1_edges)
communities = nx.community.louvain_communities(G)
communities = [sorted([i for i in j]) for j in communities]

# create community color labels for graphing
comm_colors = np.empty(len(cohort1_edges))
for i, c in enumerate(communities):
    comm_colors[np.squeeze(c)] = i
    
# connect communities to sub_cluster_labels
lc = []
for c in communities:
    lc.append(sub_cluster_labels[np.squeeze(c)])

#plot
G = nx.from_numpy_array(cohort1_edges)
nx.draw(G, nx.kamada_kawai_layout(G), node_color=comm_colors, node_size=10, edge_color='gray', width=0.25)
plt.savefig(os.path.join(results_dir, f'scaled_distance_connections_cohort1.png'), dpi=500)

#save Louvain communities (labeled by sub_cluster_labels)
lc_out = {
    'Cohort' : 1,
    'community_ids' : communities,
    'community_sub_cluster_labels' : lc,
    'community_colors' : comm_colors
}

#save results
pickle.dump(lc_out, open(os.path.join(results_dir, 'scaled_distance_Louvain_communities_cohort1.pkl'), 'wb'))
pd.DataFrame([pd.Series(c) for c in lc]).T.to_csv(os.path.join(results_dir, 'scaled_distance_Louvain_communities_sub_cluster_labels_cohort1.csv'))

#load previously saved results
#lc1 = pickle.load(open(os.path.join(results_dir, 'scaled_distance_Louvain_communities_cohort1.pkl'), 'rb'))

# Cohort 2

# isolate Cohort 2
cohort2_edges = all_edges[cohort2_start:,cohort2_start:]

# Louvain Community Detection
G = nx.from_numpy_array(cohort2_edges)
communities = nx.community.louvain_communities(G)
communities = [sorted([i for i in j]) for j in communities]

# create community color labels for graphing
comm_colors = np.empty(len(cohort2_edges))
for i, c in enumerate(communities):
    comm_colors[np.squeeze(c)] = i
    
# connect communities to sub_cluster_labels
lc = []
for c in communities:
    lc.append(sub_cluster_labels[cohort2_start:][np.squeeze(c)])

#plot
G = nx.from_numpy_array(cohort2_edges)
nx.draw(G, nx.kamada_kawai_layout(G), node_color=comm_colors, node_size=10, edge_color='gray', width=0.25)
plt.savefig(os.path.join(results_dir, f'scaled_distance_connections_cohort2.png'), dpi=500)

#save Louvain communities (labeled by sub_cluster_labels)
lc_out = {
    'Cohort' : 2,
    'community_ids' : communities,
    'community_sub_cluster_labels' : lc,
    'community_colors' : comm_colors
}

#save results
pickle.dump(lc_out, open(os.path.join(results_dir, 'scaled_distance_Louvain_communities_cohort2.pkl'), 'wb'))
pd.DataFrame([pd.Series(c) for c in lc]).T.to_csv(os.path.join(results_dir, 'scaled_distance_Louvain_communities_sub_cluster_labels_cohort2.csv'))

#load previously saved results
#lc2 = pickle.load(open(os.path.join(results_dir, 'scaled_distance_Louvain_communities_cohort2.pkl'), 'rb'))

## Summarize cluster pairs that are similar within Louvain communities
for cohort in range(2):
    print('Cohort',cohort+1)
    
    match cohort:
        case 0:
            group_idx = np.arange(0,cohort2_start)
            community_ids = lc1['community_ids']
        case 1:
            group_idx = np.arange(cohort2_start,313)
            community_ids = lc2['community_ids']
            
    #isolate cohort
    tmp = all_edges[group_idx][:,group_idx]
    
    for i, c_idx in enumerate(community_ids):
        
        #isolate community
        comm_tmp = tmp[c_idx][:,c_idx]
        
        # count edges
        c_clusters = np.sum(comm_tmp, axis=0) + np.sum(comm_tmp, axis=1)
        
        print('Community',i+1,':',len(c_idx),'pattern(s) :',
              '\n',
              'M=',np.round(np.mean(c_clusters),1),
              np.round((np.mean(c_clusters) / (len(c_clusters)-1)) * 100,1),'%',
              '\n',
              'SD=',np.round(np.std(c_clusters),1),
              np.round((np.std(c_clusters) / (len(c_clusters)-1)) * 100,1),'%',
              '\n',
              'Min=',np.round(np.min(c_clusters),1),
              np.round((np.min(c_clusters) / (len(c_clusters)-1)) * 100,1),'%',
              '\n',
              'Max=',np.round(np.max(c_clusters),1),
              np.round((np.max(c_clusters) / (len(c_clusters)-1)) * 100,1),'%')
