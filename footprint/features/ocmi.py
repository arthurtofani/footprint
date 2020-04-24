import numpy as np

# Ordered chromagram magnitude indexes
def ocmi(feature):
  return np.argsort(feature.T).T

# OCMI normalized
# uses relative distances to the most prominent index
def ocmi_norm(feature):
  ft = ocmi(feature).T
  return np.array([[(r - frame[0]) % 12 for r in frame][1:] for frame in ft]).T

# delta-ocmi measures the difference between two frames
# key-insensitive
def docmi(feature):
    r = ocmi(feature)
    arr = [[] + [__idx_dist(*values) for values in zip(A, A[1:])] for A in r]
    return np.array(arr)

def tokenize(feature, char_offset=65):
    words = [''.join([chr(a) for a in (x+char_offset)]) for x in feature.T]
    return ' '.join(words)

def __idx_dist(idx_1, idx_2):
    return (idx_2 - idx_1) % 12
