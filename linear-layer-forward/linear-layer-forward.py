import numpy as np
def linear_layer_forward(X, W, b):

    result=np.array(X)@np.array(W)+np.array(b)

    return result.tolist()

    

    
  