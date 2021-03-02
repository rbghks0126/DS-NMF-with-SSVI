import numpy as np
import scipy.stats
import time
import sys

from sklearn.base import BaseEstimator, TransformerMixin

EPS = np.spacing(1)


class Gibbs_BP_NMF(BaseEstimator, TransformerMixin):
    '''
    (Pseudo)-Collapsed Gibbs sampler for Beta process Poisson NMF
    '''

    def __init__(self, n_components=500, burn_in=500, random_state=None, n_sampling=2000,
                 verbose=False, **kwargs):
        # truncation level
        self.n_components = n_components
        # number of iterations for burn-in
        self.burn_in = burn_in
        self.random_state = random_state
        self.verbose = verbose
        # added
        self.n_sampling = n_sampling
        self.time_sampling = 0
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        # hyperparameters for components
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

        # hyperparameters for activation
        self.c = float(kwargs.get('c', 0.1))
        self.d = float(kwargs.get('d', 0.1))

        # hyperparameters for sparsity on truncated beta process
        self.a0_H = float(kwargs.get('a0_H', 1.))
        self.b0_H = float(kwargs.get('b0_H', 1.))

        # (ADDED) hyperparam for sparsity on W
        self.a0_W = float(kwargs.get('a0_W', 1.))
        self.b0_W = float(kwargs.get('b0_W', 1.))

    def fit(self, X, keep=10, threshold=None):
        ''' Do full sweep across the data for burn-in '''
        n_feats, n_samples = X.shape
        # randomly initialize parameters
        self.W = np.random.gamma(self.a, 1. / self.b,
                                 size=(n_feats, self.n_components))
        self.H = np.random.gamma(self.c, 1. / self.d,
                                 size=(self.n_components, n_samples))
        self.pi_H = np.random.beta(
            self.a0_H, self.b0_H, size=self.n_components)
        self.pi_W = np.random.beta(
            self.a0_W, self.b0_W, size=self.n_components)

        # randomly initalize binary masks for W and H
        self.S_H = (np.random.rand(self.n_components, n_samples) > .5)
        # (ADDED) initialize binary mask for W
        self.S_W = (np.random.rand(n_feats, self.n_components) > .5)

        start_time_sample = time.time()
        self._burnin(X)

        """ sampling phase """
        self.sample(X, keep, threshold)
        time_sample = time.time() - start_time_sample
        self.time_sampling = time_sample
        return self

    def sample(self, X, keep, threshold):
        """
        gibbs sampling phase after burn-in
        """
        n_feats, n_samples = X.shape
        # storage for samples from gibbs sampling
        S_W_samples = np.zeros(
            (self.n_sampling//keep, n_feats, self.n_components))
        W_samples = np.zeros(
            (self.n_sampling//keep, n_feats, self.n_components))
        S_H_samples = np.zeros(
            (self.n_sampling//keep, self.n_components, n_samples))
        H_samples = np.zeros((self.n_sampling//keep,
                              self.n_components, n_samples))
        pi_W_samples = np.zeros((self.n_sampling//keep, self.n_components))
        pi_H_samples = np.zeros((self.n_sampling//keep, self.n_components))

        if self.verbose:
            print("Sampling phase")
        for i in range(self.n_sampling):
            self.gibbs_sample(X)
            if i % keep == 0:
                # store variables
                S_W_samples[i//keep] = self.S_W.copy()
                S_H_samples[i//keep] = self.S_H.copy()
                W_samples[i//keep] = self.W.copy()
                H_samples[i//keep] = self.H.copy()
                pi_W_samples[i//keep] = self.a0_W / (self.a0_W + self.b0_W)
                pi_H_samples[i//keep] = self.a0_H / (self.a0_H + self.b0_H)

            if self.verbose and i % 20 == 0:
                sys.stdout.write('\r\tSampling iteration: %d\t' % (i))
                sys.stdout.flush()
        # take mean of sampled S_W and S_H to estimate posterior
        self.S_W_post = S_W_samples.mean(axis=0).round()
        self.S_H_post = S_H_samples.mean(axis=0).round()
        self.W_post = W_samples.mean(axis=0)
        self.H_post = H_samples.mean(axis=0)
        self.pi_W_post = pi_W_samples.mean(axis=0)
        self.pi_H_post = pi_H_samples.mean(axis=0)

        pass

    def _burnin(self, X):
        #self.log_ll = np.zeros(self.burn_in)
        if self.verbose:
            print('Gibbs burn-in')
            sys.stdout.flush()
        for b in range(self.burn_in+1):
            self.gibbs_sample(X)
            #self.log_ll[b] = self._log_likelihood(X)
            if self.verbose and b % 10 == 0:
                sys.stdout.write('\r\tIteration: %d\t' %
                                 (b))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        pass

    def gibbs_sample(self, X):
        self._gibbs_sample_S_H(X)
        self._gibbs_sample_S_W(X)
        self._gibbs_sample_WH(X)
        pass

    def _gibbs_sample_S_H(self, X):
        for k in range(self.n_components):
            # modified
            X_neg_k = (self.W * self.S_W).dot(self.H * self.S_H) - \
                np.outer(self.W[:, k] * self.S_W[:, k],
                         self.H[k] * self.S_H[k])
            # modified
            log_Ph = np.log(self.pi_H[k] + EPS) + \
                np.sum(X * np.log(X_neg_k + np.outer(self.W[:, k] * self.S_W[:, k],
                                                     self.H[k]) + EPS)
                       - np.outer(self.W[:, k] * self.S_W[:, k], self.H[k]), axis=0)
            # modified
            log_Pt = np.log(1 - self.pi_H[k] + EPS) + np.sum(X * np.log(X_neg_k + EPS),
                                                             axis=0)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_H[k] = (np.random.rand(self.S_H.shape[1]) < ratio)
        pass

    # Added gibbs sampling function for S_W
    def _gibbs_sample_S_W(self, X):
        for k in range(self.n_components):
            X_neg_k = (self.W * self.S_W).dot(self.H * self.S_H) - \
                np.outer(self.W[:, k] * self.S_W[:, k],
                         self.H[k] * self.S_H[k])
            log_Ph = np.log(self.pi_H[k] + EPS) + \
                np.sum(X * np.log(X_neg_k + np.outer(self.W[:, k],
                                                     self.H[k] * self.S_H[k]) + EPS)
                       - np.outer(self.W[:, k], self.H[k] * self.S_H[k]), axis=1)
            log_Pt = np.log(1 - self.pi_H[k] + EPS) + np.sum(X * np.log(X_neg_k + EPS),
                                                             axis=1)
            # subtract maximum to avoid overflow
            max_P = np.maximum(log_Ph, log_Pt)
            ratio = np.exp(log_Ph - max_P) / (np.exp(log_Ph - max_P) +
                                              np.exp(log_Pt - max_P))
            self.S_W[:, k] = (np.random.rand(self.S_W.shape[0]) < ratio)
        pass

    # modified
    def _gibbs_sample_WH(self, X):
        # modified
        X_hat = (self.W * self.S_W).dot(self.H * self.S_H) + EPS

        a_W = self.a + (self.W * self.S_W) * \
            (X / X_hat).dot((self.H * self.S_H).T)
        b_W = self.b + (self.H * self.S_H).sum(axis=1) * self.S_W
        self.W = np.random.gamma(a_W, 1. / b_W)

        c_H = self.c + self.H * self.S_H * (self.W * self.S_W).T.dot(X / X_hat)
        d_H = self.d + (self.W * self.S_W).sum(axis=0,
                                               keepdims=True).T * self.S_H
        self.H = np.random.gamma(c_H, 1. / d_H)

        a_pi_H = self.a0_H / self.n_components + self.S_H.sum(axis=1)
        b_pi_H = self.b0_H * (self.n_components - 1) / self.n_components \
            + self.S_H.shape[1] - self.S_H.sum(axis=1)
        self.pi_H = np.random.beta(a_pi_H, b_pi_H)

        # Added. for sampling pi_W
        a_pi_W = self.a0_W / self.n_components + self.S_W.sum(axis=0)
        b_pi_W = self.b0_W * (self.n_components - 1) / self.n_components \
            + self.S_W.shape[0] - self.S_W.sum(axis=0)
        self.pi_W = np.random.beta(a_pi_W, b_pi_W)

    def _log_likelihood(self, X):
        log_ll = 0.
        log_ll += scipy.stats.gamma.logpdf(self.W,
                                           self.a, scale=1. / self.b).sum()
        log_ll += scipy.stats.gamma.logpdf(self.H,
                                           self.c, scale=1. / self.d).sum()
        # avoid inf log-likeli
        safe_pi_H = np.maximum(self.pi_H, EPS)
        safe_pi_H = np.minimum(safe_pi_H, 1 - EPS)
        log_ll += scipy.stats.beta.logpdf(safe_pi_H, self.a0_H / self.n_components,
                                          self.b0_H * (self.n_components - 1) / self.n_components).sum()
        # added.
        safe_pi_W = np.maximum(self.pi_W, EPS)
        safe_pi_W = np.minimum(safe_pi_W, 1 - EPS)
        log_ll += scipy.stats.beta.logpdf(safe_pi_W, self.a0_W / self.n_components,
                                          self.b0_W * (self.n_components - 1) / self.n_components).sum()

        log_ll += scipy.stats.bernoulli.logpmf(self.S_H,
                                               self.pi_H[:, np.newaxis]).sum()
        # added.
        log_ll += scipy.stats.bernoulli.logpmf(self.S_W,
                                               self.pi_W[np.newaxis, :]).sum()

        log_ll += scipy.stats.poisson.logpmf(round(X),
                                             ((self.W * self.S_W).dot(self.H * self.S_H) + EPS)).sum()
        return log_ll

    def transform(self, X):
        raise NotImplementedError('Wait for it')
