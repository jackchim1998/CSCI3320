from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    plt.scatter(X[:,0], X[:,1], c=y, s=15)
    np_centers = np.array(centers)
    plt.scatter(np_centers[:,0], np_centers[:,1], c="r", s=15)
    plt.title('Ground Turth')
    plt.xlabel('value of feature 1')
    plt.ylabel('value of feature 2')
    plt.show()
    #print(y)
    # =======================================
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # you need to
    #   1. Implement the k-means by yourself
    #   and cluster samples into n_clusters clusters using your own k-means
    #
    #   2. Print out all cluster centers.
    #
    #   3. Plot all clusters formed,
    #   and use different colors to represent clusters defined by k-means.
    #   Draw a marker (e.g., a circle or the cluster id) at each cluster center.
    #
    #   4. Return scores like this: return [score, score, score, score]
    # =======================================
    num_rows, num_cols = X.shape
	#   1. Implement the k-means
    max_iteration = 500
    centers = np.zeros((n_clusters,num_cols))
    rand_arr = np.random.randint(num_rows, size=n_clusters)
    predict = np.empty((num_rows), dtype=np.int16)
    for i in range(n_clusters):
        centers[i,:] = X[rand_arr[i],:]
    for num_itertion in range(max_iteration):
        old_predict = predict.copy()
        #assign points to center
        num_points_belong_center = np.zeros((n_clusters))
        predict = np.zeros((num_rows), dtype=np.int16)
        for idx_point in range(num_rows):
            min_distance = float("inf")
            for idx_center in range(n_clusters):
                distance = 0
                for idx_feature in range(num_cols):
                    distance += (X[idx_point,idx_feature] - centers[idx_center,idx_feature]) ** 2
                if distance < min_distance:
                    min_distance = distance
                    predict[idx_point] = idx_center
            num_points_belong_center[predict[idx_point]] += 1
        #calculate new center
        centers = np.zeros((n_clusters,num_cols))
        for idx_point in range(num_rows):
            centers[predict[idx_point],:] += X[idx_point,:]
        for idx_center in range(n_clusters):
            centers[idx_center,:] = centers[idx_center,:]/num_points_belong_center[idx_center]
        if np.array_equal(old_predict, predict):
            break
    #   2. Print out all cluster centers.
    for idx_center in range(n_clusters):
        print("Center "+str(idx_center+1))
        print(centers[idx_center,:])
    #   3. Plot all clusters formed
    plt.scatter(X[:,0], X[:,1], c=predict, s=15)
    plt.scatter(centers[:,0], centers[:,1], c="r", s=15)
    plt.title('K-mean with '+str(n_clusters)+' clusters')
    plt.xlabel('value of feature 1')
    plt.ylabel('value of feature 2')
    plt.show()
    #   4. Return scores like this: return [score, score, score, score]
    ARI = metrics.adjusted_rand_score(y, predict)
    AMIS = metrics.mutual_info_score(y, predict)
    Vmeasure = metrics.v_measure_score(y, predict)
    SS = metrics.silhouette_score(X, predict, metric='euclidean')
    return [ARI,AMIS,Vmeasure,SS]  # You won't need this line when you are done

def main():
    X, y = create_dataset()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        # Implement the k-means by yourself in the function my_clustering
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
    plt.close('all')
    plt.xlabel('clusters')
    plt.ylabel('score')
    plt.title('K-mean with n clusters')
    plt.plot(range_n_clusters,ari_score,label = "ARI")
    plt.plot(range_n_clusters,mri_score,label = "MRI")
    plt.plot(range_n_clusters,v_measure_score, label = "V measure")
    plt.plot(range_n_clusters,silhouette_avg, label = "Silhouette average")
    plt.legend()
    plt.show()
    # =======================================

if __name__ == '__main__':
    main()

