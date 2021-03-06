from footprint.models import Audio
from collections import defaultdict

class Project:
  cache_folder = None
  feature_methods = None
  features_order = None
  tokenization_methods = None
  client = None
  cache_signal = False
  cache_features = False
  cache_tokens = False

  def __init__(self, cache_signal=False, cache_features=False, cache_tokens=False, cache_folder='.'):
    self.cache_folder = cache_folder
    self.feature_methods = defaultdict()
    self.tokenization_methods = defaultdict()
    self.features_order = []
    self.cache_signal = cache_signal
    self.cache_features = cache_features
    self.cache_tokens = cache_tokens

  def load_audio(self, filename):
    audio = self.__load_audio(filename)
    self.__process_features(audio)
    self.__tokenize(audio)
    audio.cleanup()
    return audio

  def set_connection(self, client):
    self.client = client

  def process_feature(self, feat_name, fn):
    self.features_order.append(feat_name)
    self.feature_methods[feat_name] = fn

  def tokenize(self, name, fn):
    self.tokenization_methods[name] = fn

  def match(self, filename, k):
    self.__assert_presence_of_client()
    audio = self.load_audio(filename)
    match_result = self.client.query(audio, k)
    return match_result

  def add(self, filename):
    self.__assert_presence_of_client()
    audio = self.load_audio(filename)
    self.client.add(audio)
    return audio

  def __process_features(self, audio):
    for f in self.features_order:
      if f not in audio.features:
        print('Loading %s for file %s...' % (f, audio.filename))
        audio.add_feature(f, self.feature_methods[f](audio))
    audio.persist()

  def __tokenize(self, audio):
    for k in self.tokenization_methods.keys():
      if k not in audio.tokens.keys():
        print('tokenizing %s' % k)
        r = self.tokenization_methods[k](audio)
        audio.add_tokens(k, r)
    audio.persist()

  def __load_audio(self, filename):
    a = Audio(filename, self)
    a.load()
    return a

  def __assert_presence_of_client(self):
    assert (self.client is not None), "Must have a client"
