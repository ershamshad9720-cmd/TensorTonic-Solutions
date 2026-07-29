import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m,n=X.shape
    #print(X.shape)
    W=np.zeros(n)
    b=0.0
    

    for _ in range(steps):
        z=X@W+b
        y_pred=_sigmoid(z)
        error=y_pred-y
        dw=(X.T@error)/m
        db=np.mean(error)
        W=(W-lr*dw)
        b=b-lr*db

    
    

    return W ,b
        