import numpy as np
#from numba import njit

#@njit
def cic(feature):
  last_frame = np.zeros(12)
  arr = []
  for frame in feature.T:
    row_d = []
    for d in np.arange(-5,7):
      sum_crm = 0
      for i in range(12):
        sum_crm = sum_crm + (last_frame[i]*frame[(i+d)%12])
      row_d.append(sum_crm)
    row_d = (row_d-np.min(row_d))/(np.max(row_d)-np.min(row_d))
    last_frame = frame
    if not np.isnan(row_d).any():
      arr.append(row_d)
  return np.array(arr).T

#@njit
def din(feature):
  arr = []
  last_frame = np.zeros(feature.shape[1])
  for frame in feature.T:
    delta = []
    for i in range(12):
      delta.append(np.linalg.norm(np.roll(frame,i) - last_frame))
    nabla = np.max(delta) - delta
    nabla = (nabla-np.min(nabla))/(np.max(nabla)-np.min(nabla))
    if not np.isnan(nabla).any():
      arr.append(nabla)
    last_frame = frame
  return np.array(arr).T
