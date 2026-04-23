# Author: Philip Deming

import os, glob
import numpy as np
import pandas as pd
import pickle
from decimal import Decimal
import matplotlib.pyplot as plt

def assign_cat_prob(noise, top_cat_idx, n_levels):
    if n_levels>2:
        x = 1 - noise
        y = noise / (n_levels - 1)
        out = np.zeros(n_levels)
        out[:] = y
        out[top_cat_idx] = x
    else:
        y = 0.5 * noise
        x = 1 - y
        out = np.zeros(n_levels)
        out[:] = y
        out[top_cat_idx] = x
    return out

def random_from_dist(M, SD):
    dist = np.random.normal(M, SD, 50000)
    out = dist[np.random.randint(1,50000)]
    return out

def divide_number_randomly(number, parts_number):
    cuts = sorted(np.random.choice(range(1,number), parts_number - 1)) #generate random cuts
    parts = []
    last_cut = 0
    for cut in cuts:
        parts.append(cut - last_cut)
        last_cut = cut
    parts.append(number - last_cut) # Add the final part

    np.random.shuffle(parts) # Shuffle the parts for more randomness
    for i in range(parts_number):
        if parts[i]<10:
            parts[parts.index(np.max(parts))]-=9 #ensure at least 10 events per cluster
            parts[i]+=9
    return parts

def simulate_clust_data(M_val, M_aro, M_RSA, M_IBI, M_PEP, M_LVET, M_SV, M_CO, main_activity, main_social, main_posture, cont_cov, noise=0, n_events=100):
    
    #affect and cardiovascular covariance matrix
    cont_cov = cont_cov * 0.025 * (1+noise) #reduce the noise and then add back systematically
    #affect and cardiovascular
    continuous = np.random.multivariate_normal([M_val, M_aro, M_RSA, M_IBI, M_PEP, M_LVET, M_SV, M_CO],
                                               cont_cov,
                                               n_events)
    continuous[:2] = np.clip(continuous[:2], -50, 50) #restrict valence/arousal to range -50 to 50
    #activity
    activity = np.random.choice(a=range(5), 
                                p=assign_cat_prob(noise, main_activity, 5), 
                                size=n_events)
    #social
    social = np.random.choice(a=range(2),
                              p=assign_cat_prob(noise, main_social, 2),
                              size=n_events)
    #posture
    posture = np.random.choice(a=range(3),
                              p=assign_cat_prob(noise, main_posture, 3),
                              size=n_events)
    
    out = pd.DataFrame(zip(continuous[:,0],continuous[:,1],continuous[:,2],continuous[:,3],
                           continuous[:,4],continuous[:,5],continuous[:,6],continuous[:,7],
                           activity, social, posture),
                       columns=['Valence','Arousal',
                                'mean_RSA','mean_IBI','mean_PEP','mean_LVET','mean_SV','mean_CO',
                                'Activity','Social.2','Posture'])
    return out

#set variables
noise = 0
n_clusters = 5
n_subjects = 20

#set paths
root_dir = 'project' #change to name of project root directory
simfiles_dir = 'simulate_files'

## Simulate data based on descriptive stats from real clusters 

for noise in [0, 0.25, 0.5, 0.75]:
    for sub in range(n_subjects):
        sub_cont_cov = pickle.load(open(os.path.join(simfiles_dir, f'covariance_subj{sub}.pkl'),'rb'))['cov']
        out=pd.DataFrame()
        n_events = divide_number_randomly(np.random.randint(low=69, high=197, size=1)[0], n_clusters)
        for j, n_events_clust in enumerate(n_events):
            if sub < 10:
                cluster_info = pickle.load(open(os.path.join(simfiles_dir, f'cluster{j}.pkl'),'rb'))
            elif 10 <= sub < 20:
                cluster_info = pickle.load(open(os.path.join(simfiles_dir, f'cluster{j+5}.pkl'),'rb'))

            #simulate
            df = simulate_clust_data(M_val=random_from_dist(cluster_info['M'][0], cluster_info['SD'][0]/10),
                                     M_aro=random_from_dist(cluster_info['M'][1], cluster_info['SD'][1]/10),
                                     M_RSA=random_from_dist(cluster_info['M'][2], cluster_info['SD'][2]/10),
                                     M_IBI=random_from_dist(cluster_info['M'][3], cluster_info['SD'][3]/10),
                                     M_PEP=random_from_dist(cluster_info['M'][4], cluster_info['SD'][4]/10),
                                     M_LVET=random_from_dist(cluster_info['M'][5], cluster_info['SD'][5]/10),
                                     M_SV=random_from_dist(cluster_info['M'][6], cluster_info['SD'][6]/10),
                                     M_CO=random_from_dist(cluster_info['M'][7], cluster_info['SD'][7]/10),
                                     cont_cov=sub_cont_cov,
                                     main_activity=int(cluster_info['act'])-1,
                                     main_social=int(cluster_info['soc'])-1,
                                     main_posture=int(cluster_info['pos'])-1,
                                     noise=noise,
                                     n_events=n_events_clust)
            df['cluster'] = j
            df['SubjectId'] = sub
            df['Activity'] = df['Activity'] + 1 #main_ari.py is looking for values 1-5
            df['Social.2'] = df['Social.2'] + 1 #main_ari.py is looking for values 1-2
            df['Posture'] = df['Posture'] + 1 #main_ari.py is looking for values 1-3
            out = pd.concat([out, df], ignore_index=True)

        #save data file
        out.to_csv(os.path.join(root_dir, f'sim_data_noise{noise}/{sub}_compiled.csv'))

        #plot and save
        vars = np.array([['Valence','Arousal','mean_RSA','mean_IBI'],['mean_PEP','mean_LVET','mean_SV','mean_CO']])
        out_M = out.groupby('cluster').mean().reset_index()
        fig, axes = plt.subplots(3,4, figsize=(8,7))
        for i in range(2):
            for j in range(4):
                out_M[vars[i][j]].plot(ax=axes[i,j], title=vars[i][j], yerr=out.groupby('cluster')[vars[i][j]].std().values, kind='bar')
        act = pd.crosstab(out['cluster'], out['Activity']).reset_index()
        soc = pd.crosstab(out['cluster'], out['Social.2']).reset_index()
        pos = pd.crosstab(out['cluster'], out['Posture']).reset_index()
        act.plot(ax=axes[2,0], y=act.keys()[1:].values, title='Activity', kind='bar')
        soc.plot(ax=axes[2,1], y=soc.keys()[1:].values, title='Social.2', kind='bar')
        pos.plot(ax=axes[2,2], y=pos.keys()[1:].values, title='Posture', kind='bar')
        fig.tight_layout()
        plt.savefig(os.path.join(root_dir, f'sim_data_noise{noise}/{sub}_sim_data_viz.png'))
        plt.close()
