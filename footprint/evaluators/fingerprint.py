import glob
import random
import os.path
import uuid

class Fingerprint:
  records_list_path = None
  db_filenames = None
  query_list_path = None
  project = None
  results = None

  def __init__(self, project):
    self.project = project

  def build(self, records_list_path):
    '''
    Receives the list of records that will be
    sent to the database.
    '''
    self.records_list_path = records_list_path
    self.db_filenames = self.get_filenames(records_list_path)
    for filename in self.db_filenames:
      print('Adding %s' % filename)
      self.project.add(filename)

  def generate_queries(self, src, amount, duration=15, queries_path=None, queries_list_path=None, results=None, sr=22050):
    '''
    This is an auxiliary method to generate queries
    from the record list.
    src: The list of filenames to create queries
    amount: how any queries to create
    duration: the query duration in seconds. When None, it uses the full record length
    '''
    import librosa

    files = self.get_filenames(src)
    os.makedirs(queries_path, exist_ok=True)

    res = []
    for i in range(amount):
      src_filename = random.choice(files)
      query_filename= self.get_query_filename(queries_path)
      offset = self.query_offset(src_filename, duration)
      y, sr = librosa.core.load(src_filename, sr=sr, duration=duration, offset=offset)
      librosa.output.write_wav(query_filename, y, sr)
      res.append([query_filename, src_filename, offset, duration])
    f = open(queries_list_path, "w")
    f.write('\n'.join(['\t'.join(k[0:1]) for k in res]))
    f.close()

    f = open(results, "w")
    f.write('\n'.join(['\t'.join(k[0:2]) for k in res]))
    f.close()
    return res

  def match(self, query_list_path):
    self.query_list_path = query_list_path
    self.results = []
    for f in self.get_filenames(query_list_path):
      res = self.project.match(f, 1)
      self.results.append(res)

  def result(self):
    return [(x[0].filename, x[1][0].filename if len(x[1])>0 else '') for x in self.results]

  def get_filenames(self, filelist):
    fl = open(filelist)
    all_files = fl.read().split('\n')
    fl.close()
    return list(set(filter(lambda f: os.path.isfile(f), all_files)))

  def __read_list(self, list_path):
    with open(list_path, 'r') as f:
      x = f.read().splitlines()
    return x

  def query_offset(self, filename, duration):
    import audioread
    with audioread.audio_open(filename) as f:
      dur = f.duration
    return random.random() * (dur - duration)

  def get_query_filename(self, queries_path):
    unique_filename = str(uuid.uuid4().hex) + '.wav'
    return os.path.join(queries_path, unique_filename)
