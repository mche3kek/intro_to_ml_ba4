"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC


class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C, kernel, gamma=1., degree=1, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """

        # Here we check that our parameter gamma is greater than 0, if it isn't we take the default value
        if (gamma < 0) :
            gamma = 1.

        # We initialise our svm using SVC and giving the corresponding parameters to the given kernel
        match kernel:
            case 'linear':
                self.svm = SVC(C, kernel)
            case 'rbf':
                self.svm = SVC(C, kernel, gamma=gamma)
            case 'poly':
                self.svm = SVC(C, kernel, degree, gamma, coef0)
                
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # We train our svm
        self.svm.fit(training_data, training_labels)
        return self.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # We use our svm to predict the right labels
        pred_labels = self.svm.predict(test_data)
        return pred_labels