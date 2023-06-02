import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # The mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # The principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        # We compute the mean of our dataset and we center it 
        self.mean = np.mean(training_data, 0)
        centered_data = training_data - self.mean

        # We compute the eigenvalues and eigenvectors of our dataset, which are used to find its principal components
        eigvals, eigvecs = np.linalg.eigh((centered_data.T@centered_data)/training_data.shape[0])

        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        self.W = eigvecs[:, :self.d]
        eg = eigvals[:self.d]

        exvar = 100*np.sum(eg) / np.sum(eigvals)

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        # We reduce the dimensionality of our data using the computed mean and principal components
        data_reduced = (data - self.mean) @ self.W
        return data_reduced
        
