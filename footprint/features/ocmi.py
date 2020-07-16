import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler

# Ordered Chromagram Magnitude Indexes
def ocmi(feature, scaled=True):
  # coefficients must be between 0..1
  r = np.argsort(1 - feature.T).T
  return (MinMaxScaler().fit_transform(r) if scaled else r)

# OCMI normalized
# uses relative distances to the most prominent index
def ocmi_rel(feature, scaled=True):
    x = ocmi(feature, False)
    r =  ((x - x[0]) % 12)[1:]
    return (MinMaxScaler().fit_transform(r) if scaled else r)

# delta-ocmi measures the difference between two frames
# key-insensitive
def docmi(feature):
  return delta(ocmi(feature))

def docmi_rel(feature):
  return delta(ocmi_rel(feature))

def delta(feature, scaled=True):
  r = librosa.feature.delta(feature)
  return (MinMaxScaler().fit_transform(r) if scaled else r)

def tokenize(feature, char_offset=65):
    words = [''.join([chr(a) for a in (x+char_offset)]) for x in feature.T]
    return ' '.join(words)
