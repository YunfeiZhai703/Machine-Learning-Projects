import scipy.linalg
import scipy.io
import numpy as np
import pandas
import matplotlib.pyplot as plt
from tqdm import tqdm

from gibbsrank import gibbs_sample
from eprank import eprank
from cw2 import sorted_barplot
from eprank import eprank

def main():
    num_iters = 1100
    data = scipy.io.loadmat('tennis_data.mat')
    names = data['W']
    G_matrix = data['G'] - 1
    samples = gibbs_sample(G_matrix, len(names), num_iters)
    
    # Select a subset of players to plot their skill evolution
    p = 5
    autocor = np.zeros(10)
    for i in range(10):
        autocor[i]=pandas.Series.autocorr(pandas.Series(samples[p,:]),lag=i)
    plt.plot(autocor)

if __name__ == "__main__":
    main()