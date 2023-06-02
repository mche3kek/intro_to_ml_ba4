import argparse

import numpy as np
import time
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    xtrain = normalize_fn(xtrain, np.mean(xtrain, axis=0), np.std(xtrain, axis=0))

    # Make a validation set
    if not args.test:
        # We use 30% of our training set to make our validation set
        k = int(0.3 * xtrain.shape[0])
        random_idx = np.random.permutation(xtrain.shape[0])
        # Use these index to extract our validation set from the training set
        xtest = xtrain[random_idx[:k]]
        ytest = ytrain[random_idx[:k]]
        xtrain = xtrain[random_idx[k:]]
        ytrain = ytrain[random_idx[k:]]

    # Dimensionality reduction (MS2)
    if args.use_pca:
        # We reshape our data to vectors
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        # We initialize our PCA object and reduce the dimensionality
        pca_obj = PCA(d=args.pca_d)
        print("Using PCA")
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        n_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":
            xtrain = xtrain.reshape(xtrain.shape[0], -1)
            xtest = xtest.reshape(xtest.shape[0], -1)
            model = MLP(input_size= xtrain.shape[1], n_classes= n_classes)  
            # We reshape our data to vectors as it is MLP's input type
            xtrain = xtrain.reshape(xtrain.shape[0], -1)
            xtest = xtest.reshape(xtest.shape[0], -1)

        elif args.nn_type == "cnn":
            xtrain = xtrain.reshape(xtrain.shape[0], 1, xtrain.shape[1], xtrain.shape[2])
            model = CNN(input_channels= xtrain.shape[1], n_classes= n_classes)
        
        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    
    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)
    

    ## 4. Train and evaluate the method

    t2 = time.time()
    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
        
    # Predict on unseen data
    preds = method_obj.predict(xtest)
    t1 = time.time()


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    # We measure the time taken to train and evaluate our algorithm
    print(f"Time = {t1 - t2:.3f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)