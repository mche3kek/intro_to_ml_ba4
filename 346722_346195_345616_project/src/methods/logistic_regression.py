import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

        

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]  # number of features
        C = training_labels.shape[1]  # number of classes
        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, training_labels) # We compute the gradient
            self.weights = self.weights - self.lr * gradient # Then compute the new weightts
        
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        probs = self.f_softmax(test_data) # Compute the probabilities
        pred_labels = onehot_to_label(probs, axis=1) # Compute the predicted label
       
        return pred_labels
    
    def f_softmax(self, data):
        """
        Softmax function
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """

        w_exp = np.exp(data @ self.weights) # Here we compute exponential of between a x vector and its corresponding weight

        exp_sum = np.sum(w_exp, axis = 1) # We compute the total of all exponentials

        exp_sum = exp_sum.reshape([exp_sum.shape[0], 1]) # We reshape it in the right shape

        y = w_exp / exp_sum # Finally we compute the probability
    
        return y
    
    def loss_logistic(self, data, labels):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        
        return -np.sum(np.log(self.f_softmax(data)) * labels) # We simply compute the loss according to the formula
    
    def gradient_logistic_multi(self, data, labels):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        
        return data.T @ (self.f_softmax(data) - labels) # We compute the gradient of E(W) for the full weight matrix