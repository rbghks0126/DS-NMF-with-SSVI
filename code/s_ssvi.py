import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import time

import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)

# original SSVI


class SSMF_BP_NMF(BaseEstimator, TransformerMixin):
    '''
    Stochastic structured mean-field variational inference for Beta process
    Poisson NMF
    '''

    def __init__(self, n_components=500, max_iter=50, burn_in=1000,
                 cutoff=1e-3, smoothness=100, random_state=None,
                 verbose=False, calc_log_ll=True, X_true=None, **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.burn_in = burn_in
        self.cutoff = cutoff
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        # ADDED
        self.calc_log_ll = calc_log_ll
        self.X_true = X_true
        self.time_list = []
        self.metrics_list = []

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        # hyperparameters for components
        self.a = float(kwargs.get('a', 5))
        self.b = float(kwargs.get('b', 5))

        # hyperparameters for activation
        self.c = float(kwargs.get('c', 5))
        self.d = float(kwargs.get('d', 5))

        # hyperparameters for sparsity on truncated beta process
        self.a0 = float(kwargs.get('a0', 1.))
        self.b0 = float(kwargs.get('b0', 1.))

        # hyperparameters for stochastic (natural) gradient
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.5))

    def _init_components(self, n_feats):
        # variational parameters for components W
        self.nu_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))
        self.rho_W = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_feats, self.n_components))

        # variational parameters for sparsity pi
        self.alpha_pi = np.random.rand(self.n_components)
        self.beta_pi = np.random.rand(self.n_components)

    def _init_weights(self, n_samples):
        # variational parameters for activations H
        self.nu_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))
        self.rho_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))

    def fit(self, X):
        # convert X to pandas
        X = pd.DataFrame.from_records(X)
        # time recording for plotting later on
        self.start_fit_time = time.time()

        n_feats, n_samples = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self.good_k = np.arange(self.n_components)
        # randomly initalize binary mask
        self.S = (np.random.rand(self.n_components, n_samples) > .5)
        self._ssmf_a(X)

        return self

    def _ssmf_a(self, X):
        # calc log_ll at the end of burn in for each iteration
        self.log_ll = np.zeros((self.max_iter))
        # ADDED
        self.good_k_list = []
        self.Epi_list = np.zeros((self.max_iter, self.n_components))
        self.Epi_H_list = np.zeros((self.max_iter, self.n_components))

        for i in range(self.max_iter):
            good_k = self.good_k
            if self.verbose:
                print('SSMF-A iteration %d\tgood K:%d:' % (i, good_k.size))
                sys.stdout.flush()
            eta = (self.t0 + i)**(-self.kappa)
            W = np.random.gamma(self.nu_W[:, good_k],
                                1. / self.rho_W[:, good_k])
            H = np.random.gamma(self.nu_H[good_k], 1. / self.rho_H[good_k])
            pi = np.random.beta(self.alpha_pi[good_k], self.beta_pi[good_k])
            for b in range(self.burn_in+1):
                # burn-in plus one actual sample
                self.gibbs_sample_S(X, W, H, pi)

                if self.calc_log_ll:
                    self.log_ll[i] = _log_likelihood(
                        X, self.S[good_k], W, H, pi)

                if self.verbose and b % 10 == 0:
                    sys.stdout.write('\r\tGibbs burn-in: %d' % b)
                    sys.stdout.flush()
            if self.verbose:
                sys.stdout.write('\n')
            self._update(eta, X, W, H)

            Epi = self.alpha_pi[good_k] / (self.alpha_pi[good_k] +
                                           self.beta_pi[good_k])
            self.good_k = good_k[Epi > Epi.max() * self.cutoff]

            # Added
            self.Epi_H_list[i] = self.alpha_pi / (self.alpha_pi +
                                                  self.beta_pi)

            self.good_k_list.append(self.good_k)
            # evaluate using current settings
            self.time_list.append(time.time() - self.start_fit_time)
            # self.metrics_list.append(
            #   evaluate(X, W.dot(H * self.S), X_true=self.X_true))

        if self.calc_log_ll == False:
            self.log_ll[i] = _log_likelihood(
                X, self.S[good_k], W, H, pi)
        pass

    def gibbs_sample_S(self, X, W, H, pi, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            X_neg_k = W.dot(H * self.S[good_k]) - np.outer(W[:, i],
                                                           H[i] * self.S[k])
            log_Ph = np.log(pi[i] + EPS) + np.sum(X * np.log(X_neg_k +
                                                             np.outer(W[:, i], H[i]) + EPS)
                                                  - np.outer(W[:, i], H[i]), axis=0)
            log_Pt = np.log(1 - pi[i] + EPS) + np.sum(X * np.log(X_neg_k + EPS),  # Added + EPS next to pi[i]
                                                      axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S[k] = (np.random.rand(self.S.shape[1]) < ratio)
            if type(log_ll) is list:
                log_ll.append(_log_likelihood(X, self.S[good_k], W, H, pi))
        pass

    def _update(self, eta, X, W, H):
        good_k = self.good_k
        X_hat = W.dot(H * self.S[good_k]) + EPS
        # update variational parameters for components W
        self.nu_W[:, good_k] = (1 - eta) * self.nu_W[:, good_k] + \
            eta * (self.a + W * (X / X_hat).dot((H * self.S[good_k]).T))
        self.rho_W[:, good_k] = (1 - eta) * self.rho_W[:, good_k] + \
            eta * (self.b + (H * self.S[good_k]).sum(axis=1))

        # update variational parameters for activations H
        self.nu_H[good_k] = (1 - eta) * self.nu_H[good_k] + \
            eta * (self.c + H * self.S[good_k] * W.T.dot(X / X_hat))
        self.rho_H[good_k] = (1 - eta) * self.rho_H[good_k] + \
            eta * (self.d + W.sum(axis=0)[:, np.newaxis] * self.S[good_k])

        # update variational parameters for sparsity pi
        self.alpha_pi[good_k] = (1 - eta) * self.alpha_pi[good_k] + \
            eta * (self.a0 / self.n_components + self.S[good_k].sum(axis=1))
        self.beta_pi[good_k] = (1 - eta) * self.beta_pi[good_k] + \
            eta * (self.b0 * (self.n_components - 1) / self.n_components
                   + self.S.shape[1] - self.S[good_k].sum(axis=1))

    def get_posterior_S(self, X, n_beta, m, burn_in, good_k, keep=10, threshold=None, use_mean=True):
        """
        get posterior mean estimate of S
        n_beta: number of beta (global variables) samples to average over. Just give it any int value if use_mean=True, as it won't be used
        m: number of observations sampled from collapsed gibbs sampling including burn-in
        burn_in: number of burn-in samples
        keep: keep only 10th sample to reduce autocorrelation
        use_mean: if true, just use the posterior mean instead of sampling multiple beta
        """
        start_time_post = time.time()

        # settings of S before running below, because gibbs sampling function changes S values
        self.store_S = self.S[good_k].copy()
        n_feats, n_samples = X.shape

        # do multiple beta sampling then averaging
        if use_mean == False:
            # 3d array containing all posterior estimates across the beta samples. for averaging across beta samples.
            S_across = np.zeros((n_beta, len(good_k), n_samples))

            for i in range(n_beta):
                print(f"beta sample: {i+1}")
                sys.stdout.flush()
                # sample W, H, and pi's
                W_sample = np.random.gamma(
                    self.nu_W[:, good_k], 1. / self.rho_W[:, good_k])
                H_sample = np.random.gamma(
                    self.nu_H[good_k], 1. / self.rho_H[good_k])
                pi_sample = np.random.beta(
                    self.alpha_pi[good_k], self.beta_pi[good_k])

                # 3d array for posterior estimates within one beta sample. for averaging samples from gibbs.
                S_within = np.zeros(((m//keep)+1, len(good_k), n_samples))

                # do gibbs sampling for burn_in + m samples
                for j in range(burn_in + m):
                    self.gibbs_sample_S(X, W_sample, H_sample, pi_sample)
                    # only keep every 10th sample to reduce autocorrelation
                    if j >= burn_in and j % keep == 0:
                        S_within[(j-burn_in) // keep] = self.S[good_k].copy()

                    # reset S as initial settings before running gibbs, as gibbs changes these
                    self.S[good_k] = self.store_S.copy()
                # ignore burn in and average entries
                S_across[i] = S_within.mean(axis=0)
            # average entries (matrices) across beta samples
            S_post = S_across.mean(axis=0)

            if threshold is None:
                S_post = S_post.round()
            else:
                S_post = np.where(S_post > threshold, 1, 0)

            time_post = time.time() - start_time_post
            self.time_post_gibbs = time_post
            self.S_post = S_post
            pass

        # dont need to do sampling for beta's. just use vi parameters mean.
        else:
            S_samples = np.zeros(((m//keep), len(good_k), n_samples))
            W_post, H_post, _ = get_post_means(self)
            # burn-in samples
            for j in range(burn_in):
                self.gibbs_sample_S(X, W_post, H_post, self.Epi_list[-1])
            for j in range(m):
                self.gibbs_sample_S(X, W_post, H_post, self.Epi_list[-1])
                if j % keep == 0:
                    S_samples[j//keep] = self.S[good_k].copy()

            time_post = time.time() - start_time_post
            self.time_post_gibs = time_post
            if threshold is None:
                self.S_post = S_samples.mean(axis=0).round()
                return self.S_post
            else:
                self.S_post = np.where(S_post > threshold, 1, 0)
                return self.S_post

    def transform(self, X):
        raise NotImplementedError('Wait for it')


def _log_likelihood(X, S, W, H, pi):
    log_ll = scipy.stats.bernoulli.logpmf(S, pi[:, np.newaxis]).sum()
    if X.dtypes[0] != int:
        log_ll += scipy.stats.poisson.logpmf(round(X),
                                             (W.dot(H * S) + EPS)).sum()
    else:
        log_ll += scipy.stats.poisson.logpmf(X, (W.dot(H * S) + EPS)).sum()
    return log_ll
