import numpy as np

def iMDL_cluster_cost(data, clusters, k=1, cont_indices=np.array([]),
                      soc_ind=np.array([8]), act_ind=np.array([9]), pos_ind=np.array([10])):

    C_size = np.sum(clusters['c_ids'] == k)
    if C_size <= 1:
        iMDL = 1e100
    else:
        cluster_cost = 0
        for j in cont_indices:
            cluster_cost += continuous_cost(clusters['sigmas'][k][j])
        if soc_ind.size > 0:
            cluster_cost += categorical_cost(clusters['p_soc'][k])
        if pos_ind.size > 0:
            cluster_cost += categorical_cost(clusters['p_pos'][k])
        if act_ind.size > 0:
            cluster_cost += categorical_cost(clusters['p_act'][k])
        cluster_cost = C_size * cluster_cost
        # if include_co:
        #     cluster_cost = C_size * (categorical_cost(clusters['p_soc'][k]) + \
        #                              categorical_cost(clusters['p_pos'][k]) + \
        #                              categorical_cost(clusters['p_act'][k]) + \
        #                              continuous_cost(clusters['sigmas'][k][0]) + \
        #                              continuous_cost(clusters['sigmas'][k][1]) + \
        #                              continuous_cost(clusters['sigmas'][k][2]) + \
        #                              continuous_cost(clusters['sigmas'][k][3]) + \
        #                              continuous_cost(clusters['sigmas'][k][4]) + \
        #                              continuous_cost(clusters['sigmas'][k][5]) + \
        #                              continuous_cost(clusters['sigmas'][k][6]) + \
        #                              continuous_cost(clusters['sigmas'][k][7])
        #                              )
        # else:
        #     cluster_cost = C_size * (categorical_cost(clusters['p_soc'][k]) + \
        #                              categorical_cost(clusters['p_pos'][k]) + \
        #                              categorical_cost(clusters['p_act'][k]) + \
        #                              continuous_cost(clusters['sigmas'][k][0]) + \
        #                              continuous_cost(clusters['sigmas'][k][1]) + \
        #                              continuous_cost(clusters['sigmas'][k][2]) + \
        #                              continuous_cost(clusters['sigmas'][k][3]) + \
        #                              continuous_cost(clusters['sigmas'][k][4]) + \
        #                              continuous_cost(clusters['sigmas'][k][5]) + \
        #                              continuous_cost(clusters['sigmas'][k][6])
        #                              )

        idc = C_size * id_cost(len(data), C_size)
        iMDL = cluster_cost + idc

    return iMDL

def iMDL_cost(data, clusters, K=2, cont_indices=np.array([]),
              soc_ind=np.array([8]), act_ind=np.array([9]), pos_ind=np.array([10])):

    iMDL = np.zeros(shape=K)
    for k in range(K):
        iMDL[k] = iMDL_cluster_cost(data, clusters, k,
                                    cont_indices=cont_indices,
                                    soc_ind=soc_ind, act_ind=act_ind, pos_ind=pos_ind)
    len_cat_attributes = []
    if soc_ind.size > 0:
        len_cat_attributes.append(len(clusters['p_soc'][k]))
    if pos_ind.size > 0:
        len_cat_attributes.append(len(clusters['p_pos'][k]))
    if act_ind.size > 0:
        len_cat_attributes.append(len(clusters['p_act'][k]))
    param_cost = (np.log2(len(data)) *
                     parameter_cost(len_cat_attributes=np.array(len_cat_attributes),
                                                  num_cont=len(cont_indices)))
    return iMDL.sum() + param_cost * K

def categorical_parameter(data, attributes_values):
    """

    :param data: points belonging to a cluster C with only attribute A
    :param attributes_values: possible values taken by A
    :return: empirical probability for attribute A for cluster C
    """
    empirical_probability = np.zeros(len(attributes_values))
    for (i, a) in enumerate(attributes_values):
        empirical_probability[i] = np.sum(data == a) / len(data)

    return empirical_probability

def categorical_cost(empirical_probability):
    """
    calculates coding cost for a categorical attribute A for a cluster C
    :param empirical_probability: empirical probability for attribute A for cluster C
    :return: coding cost of A for cluster C
    """

    low_prob_indices = np.where(empirical_probability < 1e-100)
    highest_prob_index = np.argmax(empirical_probability)
    empirical_probability[low_prob_indices] = empirical_probability[low_prob_indices] + 1e-100
    empirical_probability[highest_prob_index] = empirical_probability[highest_prob_index] \
                                                - (1e-100) * len(low_prob_indices)
    cc_a = - np.sum(empirical_probability * np.log2(empirical_probability))

    return cc_a

def gaussian_parameters(data):
    """
    estimate empirical parameters of gaussian distribution for a given cluster C
    :param data: points belongint to cluster C with only attribute B
    :return: empirical mean and variance
    """

    empirical_mean = np.mean(data)
    empirical_std = np.std(data)

    return empirical_mean, empirical_std

def continuous_cost(empirical_std):
    """
    calculates coding cost for a continous attribute B for a cluster C assuming gaussian distribution
    :param empirical_var: empirical variance of points belonging to cluster C with only attribute B
    :return: coding cost of B for cluster C
    """
    # if empirical_std == 0:
    #     empirical_std = 0.26
    if empirical_std < 0.25:
        empirical_std = 0.25
    empirical_var = empirical_std ** 2
    # if empirical_var == 0:
    #     empirical_var = 1e-300
    cc_b = 0.5 * np.log(empirical_var * 2 * np.pi * np.e) * np.log2(np.e)

    return cc_b

def parameter_cost(len_cat_attributes, num_cont,):
    """
    calculates parameter cost for cluster C
    :param len_cat_attributes: array containing number of unique values for each categorical attribute
    :param num_cont: number of continuous features
    :param cluster_size: size of cluster C
    :return: parameter cost for cluster C
    """

    cat_parameter_cost = (np.array(len_cat_attributes) - 1).sum()
    cont_parameter_cost = num_cont * 2

    pc_c = 0.5 * (cat_parameter_cost + cont_parameter_cost)

    return pc_c

def id_cost(N, cluster_size):
    """
    calculates ID cost for cluster C
    :param N: length of the data
    :param cluster_size: size of the cluster C
    :return: ID cost for cluster C
    """
    idc_c = np.log2(N / cluster_size)

    return idc_c
