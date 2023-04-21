import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        # We define K as the number of our clusters
        K = self.K

        # 1) Initialize the centers :
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:K]
        # Use these index to select centers from data
        centers = data[random_idx[:K]]

        # 2) Loop over the iterations
        for i in range(max_iter):
            # We keep in memory the centers of the previous iteration
            old_centers = centers.copy()  

            # We assign each data to its corresponding cluster
            cluster_assignments = self.assign_cluster(data)

            # We compute the new centers 
            for k in range(K):
                centers[k] = np.mean(data[cluster_assignments == k], axis=0)

            # End of the algorithm if the centers have converged
            if (np.all(old_centers == centers)): 
                break
        
        return centers, cluster_assignments
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        # We call the KMeans algorithm on our training data
        centers, cluster_assignments = self.k_means(training_data)

        # We use voting to attribute a label to each cluster center.
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(training_labels[cluster_assignments == i]))
            cluster_center_label[i] = label
    
        # We assign each data to its corresponding cluster
        cluster_assignments = self.assign_cluster(training_data)

        # Convert cluster index to label
        self.new_labels = cluster_center_label[cluster_assignments]

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        # We assign each data to its corresponding cluster
        pred_labels = self.assign_cluster(test_data)
        
        return pred_labels
    
    def assign_cluster(self, data):
        """
        Computes the Euclidean distance of each data to the clusters and assign it to the corresponding one.
        
        Arguments:
            data (np.array): data of shape (N,D)
        Returns:
            cluster_assignments (array): shape (N,) cluster assignment for each data point.
        """

        # We define N and K as the size of our data set and of our clusters respectively
        N = data.shape[0]
        K = self.K

        # Here, we will loop over the cluster
        distances = np.zeros((N, K))
        for k in range(K):
            # Compute the euclidean distance for each data to each center
            center = self.new_labels[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))

        # We assign each data to its closest cluster 
        return np.argmin(distances, axis=1)