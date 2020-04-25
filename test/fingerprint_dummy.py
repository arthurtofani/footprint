import unittest
from footprint.models import Audio
from footprint.models import Project
import footprint.clients as db
import footprint.tokenizers as tokenizers
import librosa

class TestFingerprintDummy(unittest.TestCase):
  '''
  Tests fingerprint evaluator with dummy database
  '''

  def test_creation(self):
    cache_folder = '/cache/test'
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    bucket_name = 'test'
    p = load_project()
    self.assertEqual(p.cache_folder, '/cache')

  def test_load_audio_with_cache(self):
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    p = load_project()
    audio = p.load_audio(filename)
    self.assertTrue(audio.cache_file_exists())
    audio.clean_cache()
    self.assertFalse(audio.cache_file_exists())
    audio = p.load_audio(filename)
    self.assertFalse(audio.loaded_from_cache)
    self.assertTrue(audio.cache_file_exists())
    audio = p.load_audio(filename)
    self.assertTrue(audio.loaded_from_cache)

  def test_load_audio_without_cache(self):
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    p = load_project(cache=False)
    audio = p.load_audio(filename)
    self.assertFalse(audio.loaded_from_cache)

  def test_set_connection(self):
    p = load_project(cache=False)
    set_dummy_connection(p)
    self.assertTrue(p.client.config['var1'])


  def test_add(self):
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    p = load_project(cache=True)
    set_dummy_connection(p)
    p.process_feature('chroma_cens', feat_chroma_cens)
    p.use_tokenizer('chroma_naive', naive_tokenizer_for_chroma)
    audio = p.add(filename)
    self.assertTrue(len(audio.tokens['chroma_naive']))

  def test_match(self):
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    correct_file = '/dataset/YTCdataset/letitbe/v0.mp3'
    p = load_project(cache=True)
    set_dummy_connection(p)
    p.process_feature('chroma_cens', feat_chroma_cens)
    p.use_tokenizer('chroma_naive', naive_tokenizer_for_chroma)
    p.add(correct_file)
    p.add('/dataset/YTCdataset/letitbe/v1.mp3')
    p.add('/dataset/YTCdataset/letitbe/v2.mp3')
    p.add('/dataset/YTCdataset/letitbe/v3.mp3')
    audio, results = p.match(filename, 10)
    self.assertTrue(len(audio.tokens['chroma_naive'])>0)
    self.assertTrue(results[0].filename==correct_file)

  def test_process_feature_wo_cache(self):
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    p = load_project(cache=False)
    set_dummy_connection(p)
    p.process_feature('chroma_cens', feat_chroma_cens)
    audio = p.add(filename)
    chroma = audio.features['chroma_cens']
    self.assertEqual(len(chroma), 12)
    self.assertFalse(audio.loaded_from_cache)
    audio.clean_cache()

    p = load_project(cache=True)
    set_dummy_connection(p)
    p.process_feature('chroma_cens', feat_chroma_cens)
    audio = p.add(filename)
    chroma = audio.features['chroma_cens']
    self.assertEqual(len(chroma), 12)
    self.assertFalse(audio.loaded_from_cache)

    p = load_project(cache=True)
    set_dummy_connection(p)
    p.process_feature('chroma_cens', feat_chroma_cens)
    audio = p.add(filename)
    chroma = audio.features['chroma_cens']
    self.assertEqual(len(chroma), 12)
    self.assertTrue(audio.loaded_from_cache)


def naive_tokenizer_for_chroma(audio):
  return tokenizers.naive_tokenizer(audio.features['chroma_cens'], pace=30)

def feat_chroma_cens(audio):
  print('running chroma for ', audio.filename)
  return librosa.feature.chroma_cens(audio.y, audio.sr)

def set_dummy_connection(p):
  cli = db.dummy.Connection(**{'var1': True})
  p.set_connection(cli)

def load_project(cache=True):
  return Project(cache=cache, cache_folder='/cache')

if __name__ == '__main__':
  unittest.main()

#      import code; code.interact(local=dict(globals(), **locals()))
