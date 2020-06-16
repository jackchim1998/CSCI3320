import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def logistic_func(x):

    ###################################################################
    # YOUR CODE HERE!
    # Output: logistic(x)
    ####################################################################
    L = 1 / (1 + math.e ** -x)
    return L

def train(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################
    num_pts, num_feas = X_train.shape
    w = np.random.rand(num_feas,1)
    w_change = np.zeros((num_feas,1))
    #w0 = np.random.normal()
    w0=1
    w0_change = 0
    while True:
        w_change.fill(0)
        w0_change = 0
        for idx_pt in range(num_pts):
            y = logistic_func(X_train[idx_pt].dot(w)+w0)
            w0_change += LearningRate * (y_train[idx_pt]-y)
            for idx_fea in range(num_feas):
                w_change[idx_fea][0] += LearningRate * (y_train[idx_pt]-y) * X_train[idx_pt][idx_fea]
        w0 += w0_change
        w += w_change
        sum = 0
        for idx_fea in range(num_feas):
            sum += w_change[idx_fea][0] ** 2
        sum += w0_change ** 2
        sum = sum ** 1/2
        if sum < tol:
            break
    weights = np.zeros((num_feas+1))
    weights[0] = w0
    for idx_fea in range(num_feas):
        weights[idx_fea+1] = w[idx_fea][0]
    return weights

def train_matrix(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05
    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################
    num_pts, num_feas = X_train.shape
    new_y_train = y_train.reshape((num_pts,1))
    new_X_train = np.insert(X_train, 0, np.ones((1, 1)), 1)
    w = np.random.rand(num_feas+1,1)
    w_change = np.zeros((num_feas+1,1))
    while True:
        w_change.fill(0)
        w_change = new_X_train.dot(w)
        w_change = logistic_func(w_change)
        w_change = new_y_train - w_change
        w_change = w_change.transpose()
        w_change = w_change.dot(new_X_train)
        w_change = w_change.transpose()
        w_change = LearningRate * w_change
        w += w_change
        sum = 0
        for ele in np.nditer(w_change):
            sum += ele ** 2
        sum = sum ** 1/2
        if sum < tol:
            break
    weights = np.zeros((num_feas+1))
    for idx_fea in range(num_feas+1):
        weights[idx_fea] = w[idx_fea][0]
    return weights

def predict(X_test, weights):

    ###################################################################
    # YOUR CODE HERE!
    # The predict labels of all points in test dataset.
    ####################################################################
    num_pts, num_feas = X_test.shape
    predictions = np.zeros((num_pts))
    for idx_pt in range(num_pts):
        p = weights[0]
        for idx_fea in range(num_feas):
            p += weights[1+idx_fea] * X_test[idx_pt][idx_fea]
        if p>=0.5:
            predictions[idx_pt]=1
        else:
            predictions[idx_pt]=0
    return predictions

def plot_prediction(X_test, X_test_prediction):
    X_test1 = X_test[X_test_prediction == 0, :]
    X_test2 = X_test[X_test_prediction == 1, :]
    plt.scatter(X_test1[:, 0], X_test1[:, 1], color='red')
    plt.scatter(X_test2[:, 0], X_test2[:, 1], color='blue')
    plt.show()


#Data Generation
n_samples = 1000

centers = [(-1, -1), (5, 10)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

# Experiments
print("Train: ")
w = train(X_train, y_train)
#print("Train_Matrix: ")
#w = train_matrix(X_train, y_train)
X_test_prediction = predict(X_test, w)
plot_prediction(X_test, X_test_prediction)
plot_prediction(X_test, y_test)

wrong = np.count_nonzero(y_test - X_test_prediction)
print ('Number of wrong predictions is: ' + str(wrong))
