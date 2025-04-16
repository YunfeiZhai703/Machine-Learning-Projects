# Machine Learning Project
ML projects and reports done during my final year of studies in University.
# Machine Learning Project Tasks Overview

This repository hosts a collection of three probabilistic machine‑learning mini‑projects developed during the 4F13 *Probabilistic Machine Learning* course.\
Each subsection summarises the core tasks you will find in the corresponding `./<project‑name>/` folder.

## 1 Gaussian‑Process Regression

| Task No. | Task description                                                                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **a**    | Optimise GP hyper‑parameters (length‑scale, signal & data noise) by maximising the log marginal likelihood; visualise mean predictions with 95 % confidence bands. citeturn0file0 |
| **b**    | Investigate sensitivity to different initialisations and compare model evidence for competing parameter sets. citeturn0file0                                                      |
| **c**    | Replace the RBF kernel with a periodic kernel; study how period and length‑scale affect extrapolation. citeturn0file0                                                             |
| **d**    | Draw noise‑free samples from a GP whose covariance is the product of periodic × RBF kernels; explore the effect of sample count. citeturn0file0                                   |
| **e**    | Fit two alternative GP models on 2‑D data (SE‑ARD vs. sum of SE‑ARDs) and compare squared‑residual heat‑maps & log evidence. citeturn0file0                                       |

## 2 Probabilistic Ranking of Tennis Players

| Task No. | Task description                                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a**    | Implement Gibbs sampling for the TrueSkill model; diagnose burn‑in and auto‑correlation of skill chains. citeturn0file1                            |
| **b**    | Derive and code Expectation Propagation (message passing) to obtain approximate marginal skill posteriors; study convergence speed. citeturn0file1 |
| **c**    | Compute pairwise skill‑dominance and match‑win probabilities for top ATP players using EP moments. citeturn0file1                                  |
| **d**    | Compare three probability‑estimation routes (marginal, joint Gaussian, direct samples) and discuss covariance effects. citeturn0file1              |
| **e**    | Generate player rankings with empirical averages, Gibbs sampling and EP; evaluate strengths & weaknesses of each method. citeturn0file1            |

## 3 Topic Modelling with Latent‑Dirichlet Allocation

| Task No. | Task description                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **a**    | Obtain the maximum‑likelihood estimate for a unigram language model and analyse its limitations on unseen words. citeturn0file2                                 |
| **b**    | Introduce a Dirichlet prior; derive the Bayesian predictive distribution and visualise how hyper‑parameter β shapes probabilities of rare words. citeturn0file2 |
| **c**    | Evaluate log likelihood & per‑word perplexity for a held‑out set; explain why categorical, not multinomial, likelihood is used. citeturn0file2                  |
| **d**    | Fit the Bayesian Mixture of Documents (BMM) with Gibbs sampling; inspect mixing proportions under different random initialisations. citeturn0file2              |
| **e**    | Train a 20‑topic LDA model; plot topic posteriors, word entropy, and track perplexity improvements over ML → Bayesian → BMM → LDA. citeturn0file2               |

---

### Repository layout

```
.
├── gaussian_process/
│   ├── src/           # MATLAB or Python notebooks
│   └── report.pdf
├── probabilistic_ranking/
│   ├── src/
│   └── report.pdf
├── lda_topic_modelling/
│   ├── src/
│   └── report.pdf
└── README.md          # this file
```

Feel free to copy‑edit section names or add implementation details as your codebase evolves.

