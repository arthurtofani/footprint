from footprint.models import Audio
from collections import defaultdict

class Project:
  cache_folder = None
  feature_methods = None
  features_order = None
  tokenization_methods = None
  client = None
  cache = False

  def __init__(self, cache=False, cache_folder='.'):
    self.cache_folder = cache_folder
    self.feature_methods = defaultdict()
    self.tokenization_methods = defaultdict()
    self.features_order = []
    self.cache = cache

  def load_audio(self, filename, proc_features=True, proc_tokens=True):
    audio = self.__load_audio(filename)
    if proc_features==True:
      self.__process_features(audio)
    if proc_tokens==True:
      self.__tokenize(audio)
    if self.cache:
      audio.persist()
    return audio

  def set_connection(self, client):
    self.client = client

  def process_feature(self, feat_name, fn):
    self.features_order.append(feat_name)
    self.feature_methods[feat_name] = fn

  def use_tokenizer(self, name, fn):
    self.tokenization_methods[name] = fn

  def match(self, filename, k):
    audio = self.load_audio(filename)
    match_result = self.client.query(audio, k)
    print(audio.filename)
    return match_result

  def add(self, filename):
    assert (self.client is not None), "Must have a client"
    audio = self.load_audio(filename)
    self.client.add(audio)
    return audio

  def __process_features(self, audio):
    for f in self.features_order:
      if f not in audio.features:
        print('Loading %s for file %s...' % (f, audio.filename))
        audio.features[f] = self.feature_methods[f](audio)
        audio.has_changed = True

  def __tokenize(self, audio):
    for k in self.tokenization_methods.keys():
      if k not in audio.tokens:
        r = self.tokenization_methods[k](audio)
        audio.tokens[k] = r
        #audio.has_changed = True

  def __load_audio(self, filename):
    a = Audio(filename, self)
    a.load()
    return a
