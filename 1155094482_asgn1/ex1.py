import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = np.multiply(x2, x2)
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    X = np.hstack((x1, x2, x3, x4, x5, x6))
    return X

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    mean_vector = X.mean(0)
    num_rows, num_cols = X.shape
    for i in range(num_rows):
        X[i,:] = X[i,:] - mean_vector
    cov = X.transpose().dot(X) / (num_rows-1)
    #cov = np.cov(X.transpose(),bias=False)
    eig_val, eig_vec = np.linalg.eigh(cov)
    sort_order = eig_val.argsort()[::-1]
    eig_val = eig_val[sort_order]
    eig_vec = eig_vec[:, sort_order]
    V = eig_vec
    D = eig_val
    #D = np.array([D]).transpose()
    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    return [V, D]


def main():
    np.random.seed(0)
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!
    V, D = pca(X)
    print("eigen_vectors(column vector)")
    print(V)
    print("eigen_val")
    print(D)
    #pca_sk=PCA()
    #pca_sk.fit(X)
    #print("PCA sk learn")
    #print(pca_sk.components_.transpose())
    #print(pca_sk.explained_variance_)
	#eigenval 
    num_row_V, num_col_V = V.shape
    arr_index_V=np.arange(num_col_V)
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.xlabel('index of eigenvalue')
    plt.ylabel('eigenvalue')
    plt.title('eigenvalues graph')
    plt.plot(arr_index_V,D)
    #POV
    sum=0
    POV=[]
    for eig_val in D:
        sum += eig_val
        POV.append(sum)
    for i in range(len(POV)):
        POV[i] = POV[i]/sum
    plt.subplot(122)
    plt.xlabel('index of eigenvalue')
    plt.ylabel('POV')
    plt.title('POV graph')
    plt.plot(arr_index_V,POV)

    plt.show()
    ####################################################################


if __name__ == '__main__':
    main()

