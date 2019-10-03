from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image


def softmax(y):
    max_of_rows = np.max(y, 1)
    m = np.array([max_of_rows, ] * y.shape[1]).T
    y = y - m
    y = np.exp(y)
    return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T
	
	
def load_data():
    """
    Loads the MNIST dataset. Reads the training files and creates matrices.
    :return: train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    train_truth: the matrix consisting of one 
                        hot vectors on each row(ground truth for training)
    test_truth: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """
    train_files = ['data/mnist/train%d.txt' % (i,) for i in range(10)]
    test_files = ['data/mnist/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load train data in N*D array (60000x784 for MNIST) 
    #                              divided by 255 to achieve normalization
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print "Train data array size: ", train_data.shape
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST) 
    #                             divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print "Test data array size: ", test_data.shape
    tmp = []
    for i, _file in enumerate(train_files):
        with open(_file, 'r') as fp:
            for line in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    train_truth = np.array(tmp, dtype='int')
    del tmp[:]
    for i, _file in enumerate(test_files):
        with open(_file, 'r') as fp:
            for _ in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    test_truth = np.array(tmp, dtype='int')
    print "Train truth array size: ", train_truth.shape
    print "Test truth array size: ", test_truth.shape
    return train_data, test_data, train_truth, test_truth
	
	
X_train, X_test, y_train, y_test = load_data()

# plot 5 random images from the training set
'''samples = np.random.randint(X_train.shape[0], size=5)
for i in samples:
    im = Image.fromarray(X_train[i].reshape(28,28)*255)
    plt.figure()
    plt.imshow(im)'''

X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))
print "Train truth array size (with ones): ", X_train.shape
print "Test truth array size (with ones): ", X_test.shape


def ml_softmax_train(t, X, lamda, W1init, W2init, options):
    """inputs :
      t: N x 1 binary output data vector indicating the two classes
      X: N x (D+1) input data vector with ones already added in the first column
      lamda: the positive regularizarion parameter
      winit: D+1 dimensional vector of the initial values of the parameters
      options: options(1) is the maximum number of iterations
               options(2) is the tolerance
               options(3) is the learning rate eta
    outputs :
      w: the trained D+1 dimensional vector of the parameters"""

    W1 = W1init
    W2 = W2init

    # Maximum number of iteration of gradient ascend
    _iter = options[0]

    # Tolerance
    tol = options[1]

    # Learning rate
    eta = options[2]

    Ewold = -np.inf
    costs = []
    for i in range(_iter):
        Ew, gradEw1, gradEw2 = cost_grad_softmax(W1, W2, X, t, lamda)
        # save cost
        costs.append(Ew)
        # Show the current cost function on screen
        print('Iteration : %d, Cost function :%f' % (i, Ew))

        # Break if you achieve the desired accuracy in the cost function
        if np.abs(Ew - Ewold) < tol:
            break

                
        # Update parameters based on gradient ascend
        W1 = W1 + eta * gradEw1
        W2 = W2 + eta * gradEw2

        Ewold = Ew

    return W1, W2, costs
	
	
def cost_grad_softmax(W1, W2, X, t, lamda):
    a1 = X.dot(W1.T)
    # activation function
    z = np.cos(a1)
    # add the bias to the hidden units
    z = np.hstack((np.ones((z.shape[0],1)), z))
    # y equals to a2
    y = z * W2.T
    
    s = softmax(y)
    # Compute the cost function to check convergence
    # Using the logsumexp trick for numerical stability - lec8.pdf slide 43
    max_error = np.max(y, axis=1)
    
    Ew = np.sum(t * y) - np.sum(max_error) - \
        np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - \
        (0.5 * lamda) * np.sum(np.square(W2))

    # calculate gradient for W2
    gradEw2 = (t - s).T.dot(z) - lamda * W2
    
    # remove the bias
    W2 = W2[:, 1:]
    # the derivative of the activation function
    h = -np.sin(a1)
    
    # calculate gradient for W1
    gradEw1 = ((t - s).dot(W2) * h).T.dot(X) - lamda * W1
    
    return Ew, gradEw1, gradEw2
	
	
def gradcheck_softmax(W1init, W2init, X, t, lamda):
    W1 = np.random.rand(*W1init.shape)
    W2 = np.random.rand(*W2init.shape)
    epsilon = 1e-6
    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])
    Ew, gradEw1, gradEw2 = cost_grad_softmax(W1, W2, x_sample, t_sample, lamda)
    print "gradEw1 shape: ", gradEw1.shape
    numericalGrad1 = np.zeros(gradEw1.shape)
    print "gradEw2 shape: ", gradEw2.shape
    numericalGrad2 = np.zeros(gradEw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad1.shape[0]):
        for d in range(numericalGrad1.shape[1]):
            w_tmp = np.copy(W1)
            w_tmp[k, d] += epsilon
            e_plus, _ , _ = cost_grad_softmax(w_tmp, W2, x_sample, t_sample, lamda)

            w_tmp = np.copy(W1)
            w_tmp[k, d] -= epsilon
            e_minus, _ , _  = cost_grad_softmax(w_tmp, W2, x_sample, t_sample, lamda)
            numericalGrad1[k, d] = (e_plus - e_minus) / (2 * epsilon)
    # Absolute norm
    print "The difference estimate for gradient of w1 is : ", np.max(np.abs(gradEw1 - numericalGrad1))
    
    for k in range(numericalGrad2.shape[0]):
        for d in range(numericalGrad2.shape[1]):
            w_tmp = np.copy(W2)
            w_tmp[k, d] += epsilon
            e_plus, _ , _  = cost_grad_softmax(W1, w_tmp, x_sample, t_sample, lamda)

            w_tmp = np.copy(W2)
            w_tmp[k, d] -= epsilon
            e_minus, _ , _  = cost_grad_softmax(W1, w_tmp, x_sample, t_sample, lamda)
            numericalGrad2[k, d] = (e_plus - e_minus) / (2 * epsilon)
    # Absolute norm
    print "The difference estimate for gradient of w2 is : ", np.max(np.abs(gradEw2 - numericalGrad2))
	
	
def ml_softmax_test(W1, W2, X_test):
    z = cos(X_test.dot(W1.T))
    z = np.hstack((np.ones((z.shape[0],1)), z))
    
    ytest = softmax(z.dot(W2.T))
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest
	
	
def main():
    # N of X
    N, D = X_train.shape
    M = 200
    K = 10

    # initialize w for the gradient ascent
    W1init = np.zeros((M, D))
    W2init = np.zeros((K, M+1))

    # regularization parameter
    lamda = 0.1

    # options for gradient descent
    options = [300, 1e-6, 0.5/N]

    gradcheck_softmax(W1init, W2init, X_train, y_train, lamda)

    # Train the model
    W1, W2, costs = ml_softmax_train(y_train, X_train, lamda, W1init, W2init, options)
    
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(options[2], 'f')))
    plt.show()
	
	
main()


ttest = ml_softmax_test(W, X_test)


error_count = np.not_equal(np.argmax(y_test,1), ttest).sum()
print "Error is ", error_count / y_test.shape[0] * 100, " %"