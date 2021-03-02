import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import time

import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)


class SSMF_BP_NMF(BaseEstimator, TransformerMixin):
    '''
    Stochastic structured mean-field variational inference for Beta process
    Poisson NMF
    '''
    # Added.
    # Log_ll argument. If True, calculate log likelihood.
    # X_true: if given, then evaluate using the true X_true matrix. If not, compare with given observation of X.

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
        # Added
        self.X_true = X_true
        # these record the times of respective components in algorithm
        self.calc_log_ll = calc_log_ll
        self.time_update = 0
        self.time_gibbs = 0
        self.time_ssmf = 0
        self.time_Epi = 0
        self.time_init = 0
        self.time_logll = 0
        # time of each iteration and performance measures for plotting
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
        self.a0_H = float(kwargs.get('a0_H', 1.))
        self.b0_H = float(kwargs.get('b0_H', 1.))

        ############################### (ADDED) hyperparameters for additional sparsity on W ###############################
        self.a0_W = float(kwargs.get('a0_W', 1.))
        self.b0_W = float(kwargs.get('b0_W', 1.))

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

        # variational parameters for sparsity pi_H on H matrix
        self.alpha_pi_H = np.random.rand(self.n_components)
        self.beta_pi_H = np.random.rand(self.n_components)

        ############################### (ADDED) variational parameters for sparsity pi on W matrix ###############################
        self.alpha_pi_W = np.random.rand(self.n_components)
        self.beta_pi_W = np.random.rand(self.n_components)

    def _init_weights(self, n_samples):
        # variational parameters for activations H
        self.nu_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))
        self.rho_H = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_samples))

    def fit(self, X):
        # convert X to pandas dataframe
        X = pd.DataFrame.from_records(X)
        # time recording for plotting later on
        self.start_fit_time = time.time()
        # initial evaluation just using first values before update

        n_feats, n_samples = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self.good_k = np.arange(self.n_components)
        # randomly initalize binary mask (S_H)
        self.S_H = (np.random.rand(self.n_components, n_samples) > .5)
        ############################### (ADDED) initialize S_W too ###############################
        self.S_W = (np.random.rand(n_feats, self.n_components) > .5)

        self._ssmf_a(X)
        return self

    def _ssmf_a(self, X):
        # Added. record time for ssmf
        ssmf_start_time = time.time()
        self.log_ll = np.zeros((self.max_iter))

        ############################ (ADDED)#############################
        self.good_k_list = []
        self.Epi_H_list = np.zeros((self.max_iter, self.n_components))
        self.Epi_W_list = np.zeros((self.max_iter, self.n_components))

        for i in range(self.max_iter):
            good_k = self.good_k
            if self.verbose:
                print('SSMF-A iteration %d\tgood K:%d:' % (i, good_k.size))
                sys.stdout.flush()
            eta = (self.t0 + i)**(-self.kappa)

            # Added .record time
            start_time_init = time.time()
            # initialize W, H, pi_H, pi_W
            W = np.random.gamma(self.nu_W[:, good_k],
                                1. / self.rho_W[:, good_k])
            H = np.random.gamma(self.nu_H[good_k], 1. / self.rho_H[good_k])
            pi_H = np.random.beta(
                self.alpha_pi_H[good_k], self.beta_pi_H[good_k])
            ############################### (ADDED) initialize pi_W too ###############################
            pi_W = np.random.beta(
                self.alpha_pi_W[good_k], self.beta_pi_W[good_k])
            # record time for initialization
            self.time_init += time.time() - start_time_init

            for b in range(self.burn_in+1):
                # burn-in plus one actual sample
                # record time for burn in
                start_time = time.time()
                self.gibbs_sample_S_H(X, W, H, pi_H, pi_W)
            ############################### (ADDED) gibbs sample S_W too ###############################
                self.gibbs_sample_S_W(X, W, H, pi_H, pi_W)
                t = time.time() - start_time
                self.time_gibbs += t

                start_log_ll = time.time()
                if self.calc_log_ll:
                    self.log_ll[i] = _log_likelihood(
                        X, self.S_H[good_k], self.S_W[:, good_k], W, H, pi_H, pi_W)
                self.time_logll += time.time() - start_log_ll
                if self.verbose and b % 10 == 0:
                    sys.stdout.write('\r\tGibbs burn-in: %d' % b)
                    sys.stdout.flush()
            if self.verbose:
                sys.stdout.write('\n')
            # Added. recrod time
            start_time_update = time.time()
            self._update(eta, X, W, H)
            self.time_update += time.time() - start_time_update

            start_time_Epi = time.time()
            # calculate expected pi at current iteration
            self.Epi_H_list[i] = self.alpha_pi_H / (self.alpha_pi_H +
                                                    self.beta_pi_H)
            self.Epi_W_list[i] = self.alpha_pi_W / (self.alpha_pi_W +
                                                    self.beta_pi_W)
            # only keep components with non negligible values
            alive_pi_H = good_k[self.Epi_H_list[i][good_k] >
                                self.Epi_H_list[i][good_k].max() * self.cutoff]
            alive_pi_W = good_k[self.Epi_W_list[i][good_k] >
                                self.Epi_W_list[i][good_k].max() * self.cutoff]
            self.good_k = np.array(list(set(alive_pi_H) & set(alive_pi_W)))
            ########################## (ADDED)######################################
            self.good_k_list.append(self.good_k)
            self.time_Epi += time.time() - start_time_Epi

            # evaluate using current settings
            self.time_list.append(time.time() - self.start_fit_time)
            # self.metrics_list.append(
            #   evaluate(X, (W[*self.S_W).dot(H*self.S_H), X_true=self.X_true))
        self.time_ssmf += time.time() - ssmf_start_time

        # if calc_log_ll false, just calculate log_ll at last once.
        if self.calc_log_ll == False:
            self.log_ll[i] = _log_likelihood(
                X, self.S_H[good_k], self.S_W[:, good_k], W, H, pi_H, pi_W)
        pass

    def gibbs_sample_S_H(self, X, W, H, pi_H, pi_W, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            ############################### (modified) added S_W ###############################
            # X_neg_k: F x T
            X_neg_k = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) - np.outer(W[:, i] * self.S_W[:, k],
                                                                                     H[i] * self.S_H[k])
            ############################### (modified) added S_W ###############################
            # log_Ph: 1 x T
            log_Ph = np.log(pi_H[i] + EPS) + np.sum(X * np.log(X_neg_k +
                                                               np.outer(W[:, i] * self.S_W[:, k], H[i]) + EPS)
                                                    - np.outer(W[:, i] * self.S_W[:, k], H[i]), axis=0)
            ############################### (modified) added S_W ###############################
            # log_Pt: 1 x T
            log_Pt = np.log(1 - pi_H[i] + EPS) + np.sum(X * np.log(X_neg_k + EPS),  # Added + EPS next to pi_H[i]
                                                        axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_H[k] = (np.random.rand(self.S_H.shape[1]) < ratio)

        pass

    ############################### (added) new gibbs_sample_S_W function for gibbs sampling S_W ###############################
    def gibbs_sample_S_W(self, X, W, H, pi_H, pi_W, log_ll=None):
        good_k = self.good_k
        for i, k in enumerate(good_k):
            X_neg_k = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) - np.outer(W[:, i] * self.S_W[:, k],
                                                                                     H[i] * self.S_H[k])
            log_Ph = np.log(pi_W[i] + EPS) + np.sum(X * np.log(X_neg_k +
                                                               np.outer(W[:, i], H[i] * self.S_H[k]) + EPS)
                                                    - np.outer(W[:, i], H[i] * self.S_H[k]), axis=1)

            log_Pt = np.log(1 - pi_W[i] + EPS) + np.sum(X * np.log(X_neg_k + EPS),  # Added + EPS next to pi_W[i]
                                                        axis=1)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_W[:, k] = (np.random.rand(self.S_W.shape[0]) < ratio)
        pass

    def _update(self, eta, X, W, H):
        good_k = self.good_k
        X_hat = (W * self.S_W[:, good_k]).dot(H * self.S_H[good_k]) + EPS
        # update variational parameters for components W

    ############################### (modified) add self.S_W ###############################
        self.nu_W[:, good_k] = (1 - eta) * self.nu_W[:, good_k] + \
            eta * (self.a + (W * self.S_W[:, good_k]) *
                   (X / X_hat).dot((H * self.S_H[good_k]).T))
        self.rho_W[:, good_k] = (1 - eta) * self.rho_W[:, good_k] + \
            eta * (self.b + (H * self.S_H[good_k]
                             ).sum(axis=1) * self.S_W[:, good_k])

        # update variational parameters for activations H

    ############################### (modified) W.T -> (W * S_W).T  ###############################
        self.nu_H[good_k] = (1 - eta) * self.nu_H[good_k] + \
            eta * (self.c + H * self.S_H[good_k] *
                   (W * self.S_W[:, good_k]).T.dot(X / X_hat))
        self.rho_H[good_k] = (1 - eta) * self.rho_H[good_k] + \
            eta * (self.d + (W * self.S_W[:, good_k]).sum(axis=0)
                   [:, np.newaxis] * self.S_H[good_k])

        # update variational parameters for sparsity pi_H
        self.alpha_pi_H[good_k] = (1 - eta) * self.alpha_pi_H[good_k] + \
            eta * (self.a0_H / self.n_components +
                   self.S_H[good_k].sum(axis=1))
        self.beta_pi_H[good_k] = (1 - eta) * self.beta_pi_H[good_k] + \
            eta * (self.b0_H * (self.n_components - 1) / self.n_components
                   + self.S_H.shape[1] - self.S_H[good_k].sum(axis=1))

        ############################### (added) updates for pi_W ###############################
        self.alpha_pi_W[good_k] = (1 - eta) * self.alpha_pi_W[good_k] + \
            eta * (self.a0_W / self.n_components +
                   self.S_W[:, good_k].sum(axis=0))
        self.beta_pi_W[good_k] = (1 - eta) * self.beta_pi_W[good_k] + \
            eta * (self.b0_W * (self.n_components - 1) / self.n_components
                   + self.S_W.shape[0] - self.S_W[:, good_k].sum(axis=0))
        pass

    def get_posterior_S(self, X, n_beta, m, burn_in, good_k, keep=10, threshold=None, use_mean=True):
        """
        get posterior mean estimate of S_W and S_H.
        n_beta: number of beta (global variables) samples to average over. Just give it any int value if use_mean=True, as it won't be used
        m: number of observations sampled from collapsed gibbs sampling including burn-in
        burn_in: number of burn-in samples
        keep: keep only 10th sample to reduce autocorrelation
        use_mean: if true, just use the posterior mean instead of sampling multiple beta
        """
        start_time_post = time.time()

        # settings of S_W and S_H before running below, because gibbs sampling function changes S values
        self.store_S_H = self.S_H[good_k].copy()
        self.store_S_W = self.S_W[:, good_k].copy()
        n_feats, n_samples = X.shape

        if use_mean == False:
            # 3d array containing all posterior estimates across the beta samples. for averaging across beta samples.
            S_W_across = np.zeros((n_beta, n_feats, len(good_k)))
            S_H_across = np.zeros((n_beta, len(good_k), n_samples))

            for i in range(n_beta):
                print(f"beta sample: {i+1}")
                sys.stdout.flush()
                # sample W, H, and pi's
                W_sample = np.random.gamma(
                    self.nu_W[:, good_k], 1. / self.rho_W[:, good_k])
                H_sample = np.random.gamma(
                    self.nu_H[good_k], 1. / self.rho_H[good_k])
                pi_H = np.random.beta(
                    self.alpha_pi_H[good_k], self.beta_pi_H[good_k])
                pi_W = np.random.beta(
                    self.alpha_pi_W[good_k], self.beta_pi_W[good_k])

                # 3d array for posterior estimates within one beta sample. for averaging samples from gibbs.
                S_W_within = np.zeros(((m//keep)+1, n_feats, len(good_k)))
                S_H_within = np.zeros(((m//keep)+1, len(good_k), n_samples))

                # do gibbs sampling for burn_in + m samples
                for j in range(burn_in + m):
                    self.gibbs_sample_S_H(X, W_sample, H_sample, pi_H, pi_W)
                    self.gibbs_sample_S_W(X, W_sample, H_sample, pi_H, pi_W)
                    # only keep every 10th sample to reduce autocorrelation
                    if j >= burn_in and j % keep == 0:
                        S_H_within[(j-burn_in) //
                                   keep] = self.S_H[good_k].copy()
                        S_W_within[(j-burn_in) // keep] = self.S_W[:,
                                                                   good_k].copy()
                    # reset S_H and S_W as initial settings before running gibbs, as gibbs changes these
                    self.S_H[good_k] = self.store_S_H.copy()
                    self.S_W[:, good_k] = self.store_S_W.copy()
                # ignore burn in and average entries
                S_W_across[i] = S_W_within.mean(axis=0)
                S_H_across[i] = S_H_within.mean(axis=0)
            # average entires (matrices) across beta samples
            S_W_post = S_W_across.mean(axis=0)
            S_H_post = S_H_across.mean(axis=0)

            if threshold is None:
                S_W_post = S_W_post.round()
                S_H_post = S_H_post.round()
            else:
                S_W_post = np.where(S_W_post > threshold, 1, 0)
                S_H_post = np.where(S_H_post > threshold, 1, 0)

            time_post = time.time() - start_time_post
            self.time_post_gibbs = time_post
            self.S_W_post = S_W_post
            self.S_H_post = S_H_post
            pass

        # dont need to do sampling for beta's. just use vi parameters mean.
        else:
            S_W_samples = np.zeros(((m//keep), n_feats, len(good_k)))
            S_H_samples = np.zeros(((m//keep), len(good_k), n_samples))
            W_post, H_post, _ = get_post_means(self)
            # burn-in samples
            for j in range(burn_in):
                self.gibbs_sample_S_H(
                    X, W_post, H_post, self.Epi_H_list[-1], self.Epi_W_list[-1])
                self.gibbs_sample_S_W(
                    X, W_post, H_post, self.Epi_H_list[-1], self.Epi_W_list[-1])
            for j in range(m):
                self.gibbs_sample_S_H(
                    X, W_post, H_post, self.Epi_H_list[-1], self.Epi_W_list[-1])
                self.gibbs_sample_S_W(
                    X, W_post, H_post, self.Epi_H_list[-1], self.Epi_W_list[-1])
                if j % keep == 0:
                    S_H_samples[j//keep] = self.S_H[good_k].copy()
                    S_W_samples[j//keep] = self.S_W[:, good_k].copy()
            time_post = time.time() - start_time_post
            self.time_post_gibs = time_post
            if threshold is None:
                self.S_W_post = S_W_samples.mean(axis=0).round()
                self.S_H_post = S_H_samples.mean(axis=0).round()
            else:
                self.S_W_post = np.where(S_W_post > threshold, 1, 0)
                self.S_H_post = np.where(S_H_post > threshold, 1, 0)

    def transform(self, X):
        raise NotImplementedError('Wait for it')


def _log_likelihood(X, S_H, S_W, W, H, pi_H, pi_W):
    log_ll = scipy.stats.bernoulli.logpmf(S_H, pi_H[:, np.newaxis]).sum()
    log_ll += scipy.stats.bernoulli.logpmf(S_W, pi_W[np.newaxis, :]).sum()
    if X.dtypes[0] != int:
        log_ll += scipy.stats.poisson.logpmf(round(X),
                                             ((W * S_W).dot(H * S_H) + EPS)).sum()
    else:
        log_ll += scipy.stats.poisson.logpmf(X,
                                             ((W * S_W).dot(H * S_H) + EPS)).sum()
    return log_ll
