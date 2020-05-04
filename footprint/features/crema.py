import crema
from scipy import interpolate
import numpy as np

# adapted from acoss: https://github.com/furkanyesiler/acoss/blob/master/acoss/features.py
def process(audio, hop_length=512):
  model = crema.models.chord.ChordModel()
  y, sr = audio.signal()
  data = model.outputs(y=y, sr=sr)
  fac = (float(sr) / 44100.0) * 4096.0 / hop_length
  times_orig = fac * np.arange(len(data['chord_bass']))
  nwins = int(np.floor(float(y.size) / hop_length))
  times_new = np.arange(nwins)
  interp = interpolate.interp1d(times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
  return interp(times_new)
