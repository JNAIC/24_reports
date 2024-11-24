import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.datasets import make_moons 

def plot_decision_boundary(model, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0,:].shape), cmap=plt.cm.Spectral)


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))

def load_planar_dataset():  
    np.random.seed(0)  
    X, y = make_moons(n_samples=400, noise=0.14, random_state=42)  
    y = y.reshape(1, -1)  
    return X.T, y 

def load_extra_datasets():  
    np.random.seed(1)  
    from sklearn.datasets import make_moons, make_blobs  
    X_moons, Y_moons = make_moons(n_samples=400, noise=0.2)  
    Y_moons = Y_moons.reshape(1, -1)  
    X_blobs, Y_blobs = make_blobs(n_samples=400, centers=2, random_state=5)  
    Y_blobs = Y_blobs.reshape(1, -1)  
    return (X_moons.T, Y_moons), (X_blobs.T, Y_blobs)  