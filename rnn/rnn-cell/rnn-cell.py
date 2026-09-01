import numpy as np
import math

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:

    su=W_hh@h_prev+W_xh@x_t+b_h

    #print(sum.shape)
   

    
    return np.tanh(su)
    