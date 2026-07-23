import numpy as np
import math

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    #x=np.array(x)
    #print(type(x))
    x=np.array(x)
    return 1/(1+np.exp(-x))

    
