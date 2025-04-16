# Machine Learning Project
ML projects and reports done during my final year of studies in University.

## 1 Gaussian‑Process Regression

| Task No. | Task description                                                                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **a**    | Optimise GP hyper‑parameters (length‑scale, signal & data noise) by maximising the log marginal likelihood; visualise mean predictions with 95 % confidence bands.  |
| **b**    | Investigate sensitivity to different initialisations and compare model evidence for competing parameter sets.                                                      |
| **c**    | Replace the RBF kernel with a periodic kernel; study how period and length‑scale affect extrapolation.                                                             |
| **d**    | Draw noise‑free samples from a GP whose covariance is the product of periodic × RBF kernels; explore the effect of sample count.                                  |
| **e**    | Fit two alternative GP models on 2‑D data (SE‑ARD vs. sum of SE‑ARDs) and compare squared‑residual heat‑maps & log evidence.                                       |

## 2 Probabilistic Ranking of Tennis Players

| Task No. | Task description                                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **a**    | Implement Gibbs sampling for the TrueSkill model; diagnose burn‑in and auto‑correlation of skill chains.                        |
| **b**    | Derive and code Expectation Propagation (message passing) to obtain approximate marginal skill posteriors; study convergence speed.  |
| **c**    | Compute pairwise skill‑dominance and match‑win probabilities for top ATP players using EP moments.                                |
| **d**    | Compare three probability‑estimation routes (marginal, joint Gaussian, direct samples) and discuss covariance effects.             |
| **e**    | Generate player rankings with empirical averages, Gibbs sampling and EP; evaluate strengths & weaknesses of each method.         |

## 3 Topic Modelling with Latent‑Dirichlet Allocation

| Task No. | Task description                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **a**    | Obtain the maximum‑likelihood estimate for a unigram language model and analyse its limitations on unseen words.                          |
| **b**    | Introduce a Dirichlet prior; derive the Bayesian predictive distribution and visualise how hyper‑parameter β shapes probabilities of rare words. |
| **c**    | Evaluate log likelihood & per‑word perplexity for a held‑out set; explain why categorical, not multinomial, likelihood is used.                |
| **d**    | Fit the Bayesian Mixture of Documents (BMM) with Gibbs sampling; inspect mixing proportions under different random initialisations.          |
| **e**    | Train a 20‑topic LDA model; plot topic posteriors, word entropy, and track perplexity improvements over ML → Bayesian → BMM → LDA.             |

---

