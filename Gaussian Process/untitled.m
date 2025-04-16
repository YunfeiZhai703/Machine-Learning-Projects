clear all; close all;

data = load("cw1e.mat"); x = data.x; y = data.y;

% model 1: GP with covSEard covariance
covfunc1 = @covSEard; hyp1.cov = ones(3,1); hyp1.lik = 0.1;   

% model 2: GP with {@covSum, {@covSEard, @covSEard}} covariance
covfunc2 = {@covSum, {@covSEard, @covSEard}}; hyp2.cov = 0.1 * randn(6,1); hyp2.lik = 0.1;        

% global Likelihood function
likfunc = @likGauss;

% optimization
hyp1_optimized = minimize(hyp1, @gp, -100, @infGaussLik, [], covfunc1, likfunc, x, y);
hyp2_optimized = minimize(hyp2, @gp, -100, @infGaussLik, [], covfunc2, likfunc, x, y);

% log marginal likelihoods
nlml1 = gp(hyp1_optimized, @infGaussLik, [], covfunc1, [], x, y);
nlml2 = gp(hyp2_optimized, @infGaussLik, [], covfunc2, [], x, y);

% Create a grid for prediction
[x1Grid, x2Grid] = meshgrid(linspace(min(x(:,1)), max(x(:,1)), 11), ...
                            linspace(min(x(:,2)), max(x(:,2)), 11));
xTest = [x1Grid(:), x2Grid(:)];

% Predictions for both models
[mu1, ~] = gp(hyp1_optimized, @infGaussLik, [], covfunc1, [], x, y, xTest);
[mu2, ~] = gp(hyp2_optimized, @infGaussLik, [], covfunc2, [], x, y, xTest);