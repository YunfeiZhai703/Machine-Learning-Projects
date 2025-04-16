import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming skill_samples is an array with shape (num_players, num_samples)
# We will consider just one player's samples for simplicity
# Replace with the actual player index you're interested in

def plot_autocovariance(samples, max_lag):
    # Calculate the autocovariance coefficient for a range of lags
    autocov_coeffs = [np.cov(samples[:-lag], samples[lag:])[0, 1] / np.var(samples) if lag != 0 else np.var(samples) / np.var(samples) for lag in range(max_lag + 1)]

    plt.figure(figsize=(10, 5))
    plt.stem(range(max_lag + 1), autocov_coeffs, use_line_collection=True)
    plt.xlabel('Lag')
    plt.ylabel('Autocovariance Coefficient')
    plt.title('Autocovariance Coefficient against Lag')
    plt.show()

