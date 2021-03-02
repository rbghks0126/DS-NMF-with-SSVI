
# Added helper functions
"""
get posterior mean of H, W and calculate a reconstruction of X from that
"""


def get_post_means(obj):
    nu_W = obj.nu_W
    rho_W = obj.rho_W
    nu_H = obj.nu_H
    rho_H = obj.rho_H
    good_k = obj.good_k
    W_post_mean = np.zeros(nu_W[:, good_k].shape)
    for i in range(W_post_mean.shape[0]):
        for j in range(W_post_mean.shape[1]):
            W_post_mean[i, j] = nu_W[:, good_k][i][j] / rho_W[:, good_k][i][j]

    H_post_mean = np.zeros(rho_H[good_k].shape)
    for i in range(H_post_mean.shape[0]):
        for j in range(H_post_mean.shape[1]):
            H_post_mean[i, j] = nu_H[good_k][i][j] / rho_H[good_k][i][j]

    X_post_mean = W_post_mean.dot(H_post_mean * obj.S[obj.good_k])
    return W_post_mean, H_post_mean, X_post_mean


def draw_two_matrices(X1, X2, shrink1=1, shrink2=0.3, first='matrix 1', second='matrix 2'):

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    ax1, ax2 = axes
    im1, im2 = ax1.imshow(X1, interpolation="none", aspect="auto"), \
        ax2.imshow(X2, interpolation="none", aspect="auto")
    plt.colorbar(im1, ax=ax1, shrink=shrink1)
    plt.colorbar(im2, ax=ax2, shrink=shrink2)
    ax1.title.set_text(first)
    ax2.title.set_text(second)
    plt.show()
    return


"""
evaluate function.
Calculate MSE, RMSE, RRMSE, log_rse for X and X_hat. If the true X matrix X_true is provided,
calculate using X_true and X_hat.
"""


def evaluate(X, X_hat, X_true=None):
    if X_true is None:
        mse = np.mean((X - X_hat)**2).mean()
        rmse = np.sqrt(mse)
        rrmse = np.sqrt(np.sum((X-X_hat)**2).sum() / np.sum(X**2).sum())
        log_rse = np.sqrt((np.sum((np.log(X + EPS) - np.log(X_hat + EPS))
                                  ** 2).sum()) / (np.sum((np.log(X+EPS))**2).sum()))
    else:
        mse = np.mean((X_true - X_hat)**2).mean()
        rmse = np.sqrt(mse)
        rrmse = np.sqrt(np.sum((X_true-X_hat)**2).sum() /
                        np.sum(X_true**2).sum())
        log_rse = np.sqrt((np.sum((np.log(X_true + EPS) - np.log(X_hat + EPS))
                                  ** 2).sum()) / (np.sum((np.log(X_true+EPS))**2).sum()))
    return mse, rmse, rrmse, log_rse


"""
plot the number of k across iterations
"""


def number_K(obj):
    len_good_k = [len(obj.good_k_list[i]) for i in range(len(obj.good_k_list))]
    plt.figure()
    plt.plot(len_good_k)
    plt.xticks(np.arange(0, 19+1, 1.0))
    plt.title("number of K's across iterations")
    plt.show()


"""
plot number of k and value of log_ll across iterations on the same plot
"""


def K_and_log_ll(obj):
    len_good_k = [len(obj.good_k_list[i]) for i in range(len(obj.good_k_list))]
    log_ll_list = obj.log_ll

    fig, ax = plt.subplots(1, 1)
    ax.plot(len_good_k, 'r', label="number of K")
    ax.set_yticks(np.arange(0, max(len_good_k)+1))
    ax.set_xticks(np.arange(0, len(log_ll_list), 5.0))

    ax2 = ax.twinx()
    ax2.plot(log_ll_list, label="log likelihood")
    ax2.ticklabel_format(useOffset=False, style='plain')

    plt.draw()
    fig.legend()
    print(log_ll_list)
    return


"""
plot the expected values of pi_W and pi_H across iterations
"""


def plot_pi(obj, true_pi_H=None, label=True):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    labels = np.arange(0, obj.n_components, 1)

    for i in range(obj.n_components):
        ax.plot(obj.Epi_H_list[:, i], label="pi_" + str(labels[i]))
    ax.title.set_text("Value of Pi_H's across iterations")
    if true_pi_H is not None:
        for j in range(len(true_pi_H)):
            ax.axhline(y=true_pi_H[j], color='grey', linestyle='dashed')
    if label:
        ax.legend()

    return
