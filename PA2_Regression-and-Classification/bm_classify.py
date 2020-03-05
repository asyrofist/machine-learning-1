import numpy as np



def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        rescaled_y = np.array([lab if lab > 0 else -1 for lab in y])
        for i in range(0, max_iterations):
            pred = np.dot(X, w.T) + b
            # pred[pred <= 0] = -1
            # pred[pred > 0] = 1
            err = np.asarray(rescaled_y * pred <= 0).astype(int)  # if opposite signs, then indicator is 1
            err = rescaled_y * err
            grad = np.dot(err, X)
            w += (step_size / N) * grad
            b += (step_size / N) * np.sum(err)
            

        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        for i in range(0, max_iterations):
            pred = sigmoid(np.dot(X, w) + b)
            err = pred - y
            grad = np.dot(X.T, err)

            w -= (step_size / N) * grad
            b -= (step_size / N) * np.sum(err)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = (np.dot(X, w) + b)
        preds = np.asarray(preds > 0.0).astype(int)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #

        preds = sigmoid(np.dot(X, w) + b)
        preds = np.asarray(preds > 0.5).astype(int)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds.astype(float)


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #

        def getOHE(y, C):
            if np.isscalar(y):
                N = 1
                yn=y
            else:
                N = y.shape[0]
                yn=y.astype(int)
            onehot = np.zeros([N, C])
            onehot[np.arange(N), yn] = 1.0
            return onehot


        modX=np.insert(X,[D],1,axis=1)
        labels=getOHE(y, C) # one hot encoding of y
        examples=np.zeros([0, D+1])
        wb=np.zeros([C,D+1])

        for i in range(max_iterations):
            n = np.random.choice(N)

            logval = np.dot(modX[n], wb.T)
            logval -= logval.max(-1, keepdims=True) #subtract maximum
            #NxC

            yp = np.exp(logval)  
            yp /= yp.sum(-1, keepdims=True) #probabilities

            yl = labels[n] #getOHE(y[n], C) #actual labels
         
            err = yp - yl
            #w_grad=err.T[:,:-1].dot(X[n])
            grad=np.dot(err.T.reshape(len(err),1),modX[n].reshape(1,D+1))
            #w_grad=np.dot(err.T.reshape(len(err),1),modX[n].reshape(1,D)
            #b_grad=err.T[:,-1]
            #deriv = np.dot(err.T, X)
            wb-=(step_size) * grad
        w=wb[:,:-1]
        b=wb[:,-1]
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        def encode_onehot(y, C):
            N = y.shape[0]
            onehot = np.zeros([N, C])
            onehot[np.arange(N), y.astype(int)] = 1.0
            return onehot

        for i in np.arange(max_iterations):
            logval = np.dot(X, w.T) + b
            logval -= logval.max(-1, keepdims=True)

            yp = np.exp(logval)  
            yp /= yp.sum(-1, keepdims=True)

            yl = encode_onehot(y, C)
            err = yp - yl
            deriv = np.dot(err.T, X)
            w -= (step_size / N) * deriv
            b -= (step_size / N) * err.sum(0)
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)

    def softmax(z):
        e = np.exp(z - np.max(z))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T

    z = (np.dot(w, X.T)).T + b
    pred = softmax(z)
    preds = preds + np.argmax(pred, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds