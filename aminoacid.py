import numpy as np
from config import get_config

args = get_config()
amino_acid = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_indx = {'pad': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12,
           'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'B': 21, 'O': 22, 'U': 23, 'X': 24,
           'Z': 25}


def to_onehot(seq, start=0):
    onehot = np.zeros((args.maxlen, 26), dtype=np.int32)
    l = min(args.maxlen, len(seq))
    for i in range(start, start + l):
        onehot[i, AA_indx.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot
