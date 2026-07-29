import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    
    max_len_seqs=len(max(seqs,key=len))
    #L = max_len if provided else max(len(seq) for seq in seqs) or 0
    #print(L)
    if len(seqs) ==0:
        return np.array(0,0)
    if(max_len is None):
        max_len= max_len_seqs
    for i in range(len(seqs)):
        #print(i)
        if len(seqs[i]) <=max_len:
            length=max_len-len(seqs[i])
            seqs[i]=seqs[i]+[pad_value]*length
            #print(seqs[i])
        else:
            seqs[i]=seqs[i][:max_len]


    
    
            
        
        

    #print(type(seqs))
        
    return np.array(seqs)