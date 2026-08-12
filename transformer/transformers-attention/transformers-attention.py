import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:



    scores=torch.matmul(Q,K.transpose(-2,-1))
    scores=scores/math.sqrt(Q.shape[-1])
    atten_weights=torch.softmax(scores,-1)

    #out=torch.matmul(atten_weights,V)
    #print(atten_weights)

    return torch.matmul(atten_weights,V)

    
    