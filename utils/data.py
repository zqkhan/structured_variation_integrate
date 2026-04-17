import numpy as np


def synthetic_data_intermediate(bin_thr=0.0, mu_1=np.array([-2, 2]), mu_2=np.array([2, 2]),
                       cov_1=1 * np.eye(2), cov_2=1.5 * np.eye(2), n_1=750, n_2=750, noise_1=1.0, noise_2=0.0):

    """

    :param p_a_1:
    :param mu_1:
    :param mu_2:
    :param cov_1:
    :param cov_2:
    :param n_1:
    :param n_2:
    :return:
    """


    X = np.zeros(shape=(n_1 + n_2, 3))

    y = np.zeros(shape=(n_1 + n_2))
    y[n_1:] = 1

    X[:n_1, :2] = np.random.multivariate_normal(mu_1, cov_1, n_1)
    X[n_1:, :2] = np.random.multivariate_normal(mu_2, cov_2, n_2)

    index_1 = np.where(X[:, 0] > bin_thr)[0]
    index_2 = np.where(X[:, 0] <= bin_thr)[0]

    index_1 = np.random.choice(index_1, int(len(index_1) * noise_1), replace=False)
    index_2 = np.random.choice(index_2, int(len(index_2) * noise_2), replace=False)

    X[index_1, -1] = 1
    X[index_2, -1] = 1

    return X, y


def synthetic_data_one(p_a_1=1, mu_1=np.array([-2, 2]), mu_2=np.array([2, 2]),
                       cov_1=1 * np.eye(2), cov_2=1.5 * np.eye(2), n_1=750, n_2=750):

    """

    :param p_a_1:
    :param mu_1:
    :param mu_2:
    :param cov_1:
    :param cov_2:
    :param n_1:
    :param n_2:
    :return:
    """

    p_a_2 = 1 - p_a_1

    X = np.zeros(shape=(n_1 + n_2, 3))

    y = np.zeros(shape=(n_1 + n_2))
    y[n_1:] = 1

    X[:n_1, :2] = np.random.multivariate_normal(mu_1, cov_1, n_1)
    X[n_1:, :2] = np.random.multivariate_normal(mu_2, cov_2, n_2)

    index_1 = np.random.choice(n_1, int(p_a_1 * n_1), replace=False)
    index_2 = np.random.choice(n_2, int(p_a_2 * n_2), replace=False) + n_1

    X[index_1, -1] = 1
    X[index_2, -1] = 1

    return X, y

def synthetic_data_two(p_a=[.9, .9, .9, .9, .1, .1] ,
                       mu=[np.array([-2, 2]), np.array([-3, 2]), np.array([5, 5]), np.array([-1.5, 1]),
                           np.array([-3, 1]), np.array([3.5, 1.5])],
                       cov=[.1 * np.eye(2), .1 * np.eye(2), .1 * np.eye(2),
                               .1 * np.eye(2), .1 * np.eye(2), .1 * np.eye(2)],
                       num_points=[200, 100, 200, 200, 100, 200]):

    """

    :param p_a:
    :param mu:
    :param cov:
    :param num_points:
    :return:
    """
    X = []
    y = []
    for (i, n) in enumerate(num_points):
        cluster_data = np.zeros(shape=(n, 3))
        cluster_data[:, :2] = np.random.multivariate_normal(mu[i], cov[i], n)
        index = np.random.choice(n, int(p_a[i] * n), replace=False)
        cluster_data[index, 2:] = 1
        X.append(cluster_data)
        y.append(i * np.ones(shape=n,))

    X = np.vstack(X)
    y = np.hstack(y)


    return X, y