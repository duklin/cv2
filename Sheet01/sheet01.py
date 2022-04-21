from cgi import test
import numpy as np
from numpy.linalg import inv

##############################################################################################################
# Auxiliary functions for Regression
##############################################################################################################
# returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)


def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:, :2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:, 2:]), axis=1)
    return Y, X


def lin_reg(X, Y):
    """Linear Regression

    Arguments
    ---------
    - X: features with bias X (num_samples*(1+num_features))
    - Y: target values Y (num_samples*target_dims)

    Returns
    -------
    - w: regression coefficients w ((1+num_features)*target_dims)
    """
    w = inv(X.T @ X) @ X.T @ Y
    return w


def test_lin_reg(X, Y, w):
    """Test the predictions from the learned Linear Regressor

    Arguments
    ---------
    - X: features with bias X (num_samples*(1+num_features))
    - Y: target values Y (num_samples*target_dims)
    - w: regression coefficients w ((1+num_features)*target_dims)

    Returns
    -------
    fraction of mean square error and variance of target prediction separately for each target dimension
    """
    Y_hat = X @ w
    rmse = np.sqrt(np.sum(np.square(Y - Y_hat), 0))
    var_y = np.square(np.std(Y, 0))

    return np.divide(rmse, var_y)


def RBF_embed(X, C, sigma):
    """Generate the Kernel Matrix for Radial Basis functions

    Arguments
    ---------
    - X: features with bias X (num_samples*(1+num_features))
    - C: centers of clusters C (num_clusters*(1+num_features))
    - sigma: standard deviation of RBF

    Returns
    -------
    - kernel: matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)
    """
    kernel = np.zeros((X.shape[0], C.shape[0]))
    for j in range(kernel.shape[1]):
        kernel[:, j] = np.exp(-0.5 * np.diag((X - C[j]) @
                              (X - C[j]).T) / (sigma**2))

    return kernel


############################################################################################################
# Linear Regression
############################################################################################################
def run_lin_reg(X_tr, Y_tr, X_te, Y_te):
    w = lin_reg(X_tr, Y_tr)
    err = test_lin_reg(X_te, Y_te, w)
    print("MSE/Var linear regression")
    print(err, "\n")


############################################################################################################
# Dual Regression
############################################################################################################
def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    opt_sigma = None
    opt_psi = None
    best_err_val = np.inf
    X_val = X_tr[val_list]
    Y_val = Y_tr[val_list]
    X_tr = X_tr[tr_list]
    Y_tr = Y_tr[tr_list]
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)
        K_tr = RBF_embed(X_tr, X_tr, sigma)
        psi = inv(K_tr) @ Y_tr
        K_val = RBF_embed(X_val, X_tr, sigma)
        err_dual = test_lin_reg(K_val, Y_val, psi)
        print("MSE/Var dual regression for val sigma=" + str(sigma))
        print(err_dual)
        if np.linalg.norm(err_dual) < best_err_val:
            best_err_val = np.linalg.norm(err_dual)
            opt_sigma = sigma
            opt_psi = psi

    K_te = RBF_embed(X_te, X_tr, opt_sigma)
    err_dual = test_lin_reg(K_te, Y_te, opt_psi)
    print("MSE/Var dual regression for test sigma=" + str(opt_sigma))
    print(err_dual, "\n")
    print("The validation set proposed in the template is not ideal as it does not randomize the split, which can lead to a bias in either of the datasets.")
    print("When sigma approaches zero, the diagonal elements of the kernel matrix are undefined due to 0/0 operation in the kernel function evaluation.")
    print("When sigma approaches infinity, the kernel matrix is all ones, which makes it rank 1 and hence singular.")
    print("In both these cases, the gradient 'psi' cannot be computed\n\n")
############################################################################################################
# Non Linear Regression
############################################################################################################


def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    from sklearn.cluster import KMeans
    best_err_val = np.inf
    opt_num_clusters = 0
    opt_sigma = 0
    opt_w = None
    for num_clusters in [10, 30, 100]:
        C_tr = (KMeans(num_clusters).fit(X_tr[tr_list])).cluster_centers_
        C_val = (KMeans(num_clusters).fit(X_tr[val_list])).cluster_centers_
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)
            K_tr = RBF_embed(X_tr[tr_list], C_tr, sigma)
            w = lin_reg(K_tr, Y_tr[tr_list])
            K_val = RBF_embed(X_tr[val_list], C_val, sigma)
            err_val = test_lin_reg(K_val, Y_tr[val_list], w)
            print(
                "MSE/Var non linear regression for val sigma="
                + str(sigma)
                + " val num_clusters="
                + str(num_clusters)
            )
            print(err_val)
            if np.linalg.norm(err_val) < best_err_val:
                best_err_val = np.linalg.norm(err_val)
                opt_num_clusters = num_clusters
                opt_sigma = sigma
                opt_w = w

    print(
        "MSE/Var non linear regression for test sigma="
        + str(opt_sigma)
        + " test num_clusters="
        + str(opt_num_clusters)
    )
    C_te = (KMeans(opt_num_clusters).fit(X_te)).cluster_centers_
    K_te = RBF_embed(X_te, C_te, opt_sigma)
    err_test = test_lin_reg(K_te, Y_te, opt_w)
    print(err_test, "\n\n")


####################################################################################################################################
# Auxiliary functions for classification
####################################################################################################################################
# returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [("bottle", 1), ("horse", 0)]:
        feat[category[0]] = np.loadtxt(
            "data/" + category[0] + "_" + split + ".txt")
        feat[category[0]] = np.concatenate(
            (np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1
        )
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat["bottle"], feat["horse"]), axis=0)
    Y = np.concatenate((gt["bottle"], gt["horse"]), axis=0)
    return Y, X


# takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1, 1)
# Y must be from {0, 1}
# returns gradient with respect to w (num_features)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_llkhd_grad(X, Y, w):
    Y_hat = sigmoid(X @ w)
    gradient = (Y_hat - Y) * X
    gradient = gradient.sum(axis=0).reshape(-1, 1)
    return gradient


# takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1, 1)
# Y must be from {0, 1}
# returns log likelihood loss
def get_loss(X, Y, w):
    Z = X @ w
    L = Y * -np.log(sigmoid(Z))
    L += (1 - Y) * -np.log(1 - sigmoid(Z))
    return L.sum()


# takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1, 1)
# Y must be from {0, 1}
# returns accuracy
def get_accuracy(X, Y, w):
    Y_hat = sigmoid(X @ w)
    Y_hat[Y_hat >= 0.5] = 1
    Y_hat[Y_hat < 0.5] = 0
    return 1 - np.count_nonzero(Y_hat - Y) / Y_hat.shape[0]


####################################################################################################################################
# Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
    print("classification with step size " + str(step_size))
    max_iter = 10000
    w = np.random.randn(X_tr.shape[1], 1)
    for step in range(max_iter):
        loss = get_loss(X_tr, Y_tr, w)
        accuracy = get_accuracy(X_te, Y_te, w)
        grad = log_llkhd_grad(X_tr, Y_tr, w)
        w -= step_size * grad
        if step % 1000 == 0:
            print(
                "step="
                + str(step)
                + " loss="
                + str(loss)
                + " accuracy="
                + str(accuracy)
            )

    print("test set loss=" + str(loss) + " accuracy=" + str(accuracy))


####################################################################################################################################
# Exercises
####################################################################################################################################
Y_tr, X_tr = read_data_reg("data/regression_train.txt")
Y_te, X_te = read_data_reg("data/regression_test.txt")

run_lin_reg(X_tr, Y_tr, X_te, Y_te)

tr_list = list(range(0, int(X_tr.shape[0] / 2)))
val_list = list(range(int(X_tr.shape[0] / 2), X_tr.shape[0]))

run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)

step_size = 0.1
Y_tr, X_tr = read_data_cls("train")
Y_te, X_te = read_data_cls("test")
Y_tr = Y_tr.reshape(-1, 1)
Y_te = Y_te.reshape(-1, 1)
run_classification(X_tr, Y_tr, X_te, Y_te, step_size)
print("The Logistic Regression model for learning linear classifier cannot get stuck in a local minima. This is because the corresponding loss landscape is a convex function, which is indicated by the positive definiteness of the Hessian of the loss function wrt the learnable parameters")
