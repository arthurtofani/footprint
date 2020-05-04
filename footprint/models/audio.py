from collections import defaultdict
import librosa
import os
import numpy as np
import h5py

class Audio:
  filename = None
  project = None
  bucket = None
  tempo = None
  beats = None
  features = None
  tokens = None
  loaded_from_cache = False
  has_changed = False

  def __init__(self, filename, project):
    self.filename = filename
    self.project = project
    self.features = defaultdict()
    self.tokens = defaultdict()
    self.signal_has_changed = False
    self.feature_has_changed = False
    self.y = None
    self.sr = None

  def load(self):
    if self.project.cache_features:
      self.__load_features_from_cache()

  def add_feature(self, feature_name, feature):
    self.features[feature_name] = feature
    self.feature_has_changed = True

  def add_tokens(self, tokens_key, tokens):
    self.tokens[tokens_key] = tokens

  def persist(self):
    if self.project.cache_features and self.feature_has_changed:
      self.persist_features()
    if self.project.cache_signal and self.signal_has_changed:
      self.persist_signal()

  def signal(self):
    if self.y==None:
      self.y, self.sr = self.__load_signal()
    return (self.y, self.sr)

  def cleanup(self):
    self.y = None
    self.sr = None

  def persist_features(self):
    self.__create_cache_folder()
    print('dumping features', self.filename)
    with h5py.File(self.__cache_filename('features'), "w") as f:
      for key in self.features.keys():
        f.create_dataset(key, data=self.features[key])
    self.feature_has_changed = False

  def persist_signal(self):
    self.__create_cache_folder()
    print('dumping audio', self.filename)
    with h5py.File(self.__cache_filename('audio'), "w") as f:
      f.create_dataset('y', data=self.y)
      f.attrs["sr"] = self.sr
    self.signal_has_changed = False

  def clean_cache(self, file_type_str):
    if self.__cache_filename_exists():
      os.remove(self.__cache_filename(file_type_str))

  def __load_signal(self):
    return self.__load_signal_from_cache() or self.__load_signal_from_file()

  def __load_signal_from_file(self):
    print('loading signal from file - %s' % self.filename)
    self.y, self.sr = librosa.load(self.filename)
    self.signal_has_changed = True
    return (self.y, self.sr)

  def __load_signal_from_cache(self):
    if not self.__cache_filename_exists('audio'):
      return None
    print('loading signal from cache - %s' % self.filename)
    with h5py.File(self.__cache_filename('audio'), 'r') as f:
      self.y  = np.array(f['y'])
      self.sr = f.attrs["sr"]
    return (self.y, self.sr)

  def __load_features_from_cache(self):
    if not self.__cache_filename_exists('features'):
      return
    with h5py.File(self.__cache_filename('features'), 'r') as f:
      for k in f.keys():
        self.features[k] = np.array(f[k])

  def __cache_filename(self, file_type_str):
    return self.__cache_folder() + ('/%s.hdf5' % file_type_str)

  def __cache_filename_exists(self, file_type_str):
    return os.path.isfile(self.__cache_filename(file_type_str))

  def __create_cache_folder(self):
    os.makedirs(self.__cache_folder(), exist_ok=True)

  def __cache_folder(self):
    fld = self.project.cache_folder
    return fld + self.filename
