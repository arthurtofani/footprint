from collections import defaultdict
import librosa
import os
import numpy as np
import h5py

class Audio:
  filename = None
  project = None
  bucket = None
  y = None
  sr = None
  tempo = None
  beats = None
  features = None
  tokens = None
  loaded_from_cache = False
  has_changed = False


  def __init__(self, filename, project, bucket=None):
    self.filename = filename
    self.project = project
    self.bucket = bucket
    self.features = defaultdict()
    self.tokens = defaultdict()
    self.has_changed = False

  def load(self):
    if self.project == None or self.project.cache==False:
      self.__load_from_file()
      self.persist()
    else:
      self.__load_from_cache()

  def persist(self):
    if self.has_changed == False:
      return
    print('dumping', self.filename)
    self.__create_cache_folder()
    with h5py.File(self.cache_filename(), "w") as f:
      f.create_dataset("y", data=self.y)
      f.create_dataset("beats", data=self.beats)
      f.attrs["sr"] = self.sr
      f.attrs["tempo"] = self.tempo
      for feat in self.features.keys():
        f.create_dataset(('features/%s' % feat), data=self.features[feat])
      #for tok in self.tokens.keys():
        #if 'tokens' not in f:
          #f.create_group('tokens')
        #if tok in f['tokens']:
          #del f['tokens'][tok]
        #f['tokens'][tok] = self.tokens[tok]

  def clean_cache(self):
    if self.cache_file_exists():
      os.remove(self.cache_filename())

  def cache_filename(self):
    return self._cache_folder() + '/audio.hdf5'

  def cache_file_exists(self):
    return os.path.isfile(self.cache_filename())

  def __load_from_file(self):
    self._load_signal_from_file()
    self.load_beats()
    self.loaded_from_cache = False
    self.has_changed = True

  def __load_from_cache(self):
    try:
      with h5py.File(self.cache_filename(), "r") as f:
        self.sr = f.attrs['sr']
        self.y = np.array(f['y'])
        self.tempo = f.attrs['tempo']
        self.beats = np.array(f['beats'])

        if 'features' in f:
          for k in f['features'].keys():
            self.features[k] = np.array(f['features'][k])
        #if 'tokens' in f:
          #for k in f['tokens'].keys():
            #self.tokens[k] = f['tokens'][k].value
      self.loaded_from_cache = True
    except OSError:
      self.__load_from_file()


  def __create_cache_folder(self):
    os.makedirs(self._cache_folder(), exist_ok=True)

  def _cache_folder(self):
    fld = self.project.cache_folder
    return fld + self.filename

  def _load_signal_from_file(self):
    self.y, self.sr = librosa.load(self.filename)

  def load_beats(self):
    self.tempo, self.beats = librosa.beat.beat_track(y=self.y, sr=self.sr, trim=False)
