from audioop import rms
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
    - Y: takes features with bias X (num_samples*(1+num_features))

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
    - Y: takes features with bias X (num_samples*(1+num_features))
    - w: regression coefficients w ((1+num_features)*target_dims)

    Returns
    -------
    fraction of mean square error and variance of target prediction separately for each target dimension
    """
    Y_hat = X @ w
    rmse = np.sqrt(np.sum(np.square(Y - Y_hat), 0))
    var_y = np.square(np.std(Y, 0))

    return np.divide(rmse, var_y)

    # takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and std of RBF sigma
    # returns matrix with scalar product values of features and cluster centers in higher embedding space (num_samples*num_clusters)


# def RBF_embed(X, C, sigma):

############################################################################################################
# Linear Regression
############################################################################################################
def run_lin_reg(X_tr, Y_tr, X_te, Y_te):
    w = lin_reg(X_tr, Y_tr)
    err = test_lin_reg(X_te, Y_te, w)
    print('MSE/Var linear regression')
    print(err)


############################################################################################################
# Dual Regression
############################################################################################################
def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)
        print('MSE/Var dual regression for val sigma='+str(sigma))
        print(err_dual)

    print('MSE/Var dual regression for test sigma='+str(opt_sigma))
    print(err_dual)

############################################################################################################
# Non Linear Regression
############################################################################################################


def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    from sklearn.cluster import KMeans
    for num_clusters in [10, 30, 100]:
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)
            print('MSE/Var non linear regression for val sigma=' +
                  str(sigma)+' val num_clusters='+str(num_clusters))
            print(err_dual)

    print('MSE/Var non linear regression for test sigma=' +
          str(opt_sigma)+' test num_clusters='+str(opt_num_clusters))
    print(err_dual)

####################################################################################################################################
# Auxiliary functions for classification
####################################################################################################################################
# returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)


def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]:
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate(
            (np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

# takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
# returns gradient with respect to w (num_features)


# def log_llkhd_grad(X, Y, w):

    # takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
    # Y must be from {-1, 1}
    # returns log likelihood loss


# def get_loss(X, Y, w):

    # takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
    # Y must be from {-1, 1}
    # returns accuracy


# def get_accuracy(X, Y, w):

    ####################################################################################################################################
    # Classification
    ####################################################################################################################################


def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
    print('classification with step size '+str(step_size))
    max_iter = 10000
    for step in range(max_iter):
        if step % 1000 == 0:
            print('step='+str(step)+' loss=' +
                  str(loss)+' accuracy='+str(accuracy))

    print('test set loss='+str(loss)+' accuracy='+str(accuracy))


####################################################################################################################################
# Exercises
####################################################################################################################################
Y_tr, X_tr = read_data_reg('data/regression_train.txt')
Y_te, X_te = read_data_reg('data/regression_test.txt')

run_lin_reg(X_tr, Y_tr, X_te, Y_te)

# tr_list = list(range(0, int(X_tr.shape[0]/2)))
# val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]))

# run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
# run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)

# step_size = 0.0001
# Y_tr, X_tr = read_data_cls('test')
# Y_te, X_te = read_data_cls('test')
# run_classification(X_tr, Y_tr, X_te, Y_te, step_size)
