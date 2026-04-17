import numpy as np

def gaussian_pdf(x, mean=0, sigma = 1):

    return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)