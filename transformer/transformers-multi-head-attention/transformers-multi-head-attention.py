import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:

    Batch_size,Seq_len,d_model=Q.shape
    d_k=d_model//num_heads
    Q=Q@W_q
    K=K@W_k
    V=V@W_v
    Q=Q.reshape(Batch_size,Seq_len,num_heads,d_k).transpose(0,2,1,3)
    K=K.reshape(Batch_size,Seq_len,num_heads,d_k).transpose(0,2,1,3)
    V=V.reshape(Batch_size,Seq_len,num_heads,d_k).transpose(0,2,1,3)
    scores=Q@K.transpose(0,1,3,2)
    scores=scores/np.sqrt(d_k)

    atten_weights=softmax(scores,-1)

    out=atten_weights@V

    out=out.transpose(0,2,1,3).reshape(Batch_size,Seq_len,d_model)

    return out@W_o
   