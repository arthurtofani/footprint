import numpy as np
from datasketch import MinHash
from collections import deque
import hashlib

def naive_tokenizer(feature, offset=70, pace=10):
  f = np.array(feature)
  return ' '.join([''.join([chr(int(i*pace)+offset) for i in fr]) for fr in f.T])

def shingle(tokens, size):
  return ['.'.join(x) for x in list(__shingle(tokens.split(), size))]

def __shingle(sequence, size):
    """
    Generator that yields shingles (a rolling window) of a given size.
    Parameters
    ----------
    sequence : iterable
               Sequence to be shingled
    size : int
           size of shingle (window)
    """
    iterator = iter(sequence)
    init = (next(iterator) for _ in range(size))
    window = deque(init, maxlen=size)
    if len(window) < size:
        raise IndexError('Sequence smaller than window size')
    yield np.asarray(window)
    for elem in iterator:
        window.append(elem)
        yield np.asarray(window)

def reshape(feature, start_index=None, end_index=None):
    return feature[start_index:end_index]

def min_hash(word_list, k=12):
  def to_hash(word):
    m = MinHash(num_perm=k)
    m.update(word.encode('utf-8'))
    return m.hashvalues
  return [to_hash(m) for m in word_list]

def to_words(min_hash_response, max_str_size=70):
  #import code; code.interact(local=dict(globals(), **locals()))
  return ' '.join([hashlib.md5('$'.join([str(x) for x in s]).encode('utf-8')).hexdigest()[:max_str_size] for s in min_hash_response])
  #return ' '.join([''.join([encode(int(x)) for x in r]) for r in min_hash_response])

BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def encode(num, alphabet=BASE62):
  if num == 0:
      return alphabet[0]
  arr = []
  base = len(alphabet)
  while num:
      num, rem = divmod(num, base)
      arr.append(alphabet[rem])
  arr.reverse()
  return ''.join(arr)

def magic_hash(feature, min_hash_fns=10, shingle_size=3):
  #return to_words(min_hash(shingle(naive_tokenizer(feature), shingle_size), min_hash_fns), 16)
  return to_words(shingle(naive_tokenizer(feature), shingle_size), 6)

