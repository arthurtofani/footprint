import unittest
import numpy as np
from footprint.models import Audio
from footprint.models import Project

class TestProject(unittest.TestCase):
  def test_process_feature(self):
    p = Project('/cache')
    p._load_audio('/dataset/YTCdataset/letitbe/test.mp3')
    p.process_feature('test', double_signal)



def double_signal(audio):
  return np.array(audio.y) * 2

if __name__ == '__main__':
    unittest.main()

#import code; code.interact(local=dict(globals(), **locals()))
