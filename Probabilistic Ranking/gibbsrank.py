import scipy.linalg
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def gibbs_sample(G, M, num_iters):
    running_mean = np.zeros((M, num_iters))
    running_variance = np.zeros((M, num_iters))
    # number of games
    N = G.shape[0]
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)
    # number of iterations of Gibbs
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N):

            s = w[G[g, 0]] - w[G[g, 1]]  # difference in skills
            t[g] = s + np.random.randn()  # Sample performance
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((M, 1))
        for p in range(M):
            m[p] =  sum(t[np.where(G[:, 0] == p)]) - sum(t[np.where(G[:, 1] == p)])
        iS = np.zeros((M, M))  # Container for sum of precision matrices (likelihood terms)

        for g in range(N):
            iS[G[g, 0], G[g, 0]] += 1
            iS[G[g, 1], G[g, 1]] += 1
            iS[G[g, 0], G[g, 1]] -= 1
            iS[G[g, 1], G[g, 0]] -= 1

        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]
        # Update running mean and variance

        running_mean[:, i] = np.mean(skill_samples[:, :i+1], axis=1)
        running_variance[:, i] = np.var(skill_samples[:, :i+1], axis=1)

        # Plot running mean and variance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(running_mean[:10, :].T)
    plt.title('Evolution of mean of Skills over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Mean')

    plt.subplot(1, 2, 2)
    plt.plot(running_variance[:10, :].T)
    plt.title('Evolution of variance of Skills over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.show()
    return skill_samples, running_mean, running_variance


