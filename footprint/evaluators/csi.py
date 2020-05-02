import numpy as np
import glob
import random
from collections import defaultdict
import librosa
import pandas as pd
import sklearn
import audioread
import os.path
from footprint import util as util
import uuid
import csv

class CSI:
  records_list_path = None
  db_filenames = None
  query_list_path = None
  project = None
  results = None
  amnt_results = 10

  def __init__(self, project):
    self.project = project

  def build(self, records_list_path, exclude_path=None):
    '''
    Receives the list of records that will be
    sent to the database.
    '''
    self.records_list_path = records_list_path
    exclude_files = []
    if exclude_path is not None:
      exclude_files  = self.get_filenames(exclude_path)
    records_files = self.get_filenames(records_list_path)
    total_record_files = [x for x in records_files if x not in exclude_files]
    self.db_filenames = total_record_files
    for filename in self.db_filenames:
      print('Adding %s' % filename)
      self.project.add(filename)


  def match(self, query_list_path):
    '''
    Performs match between the songs in the query_list_path file and those ones
    received by the #build method
    '''
    self.query_list_path = query_list_path
    self.results = []
    ct = 0
    self.queries = self.get_filenames(self.query_list_path)
    sz = len(self.queries)
    for f in self.queries:
      ct+=1
      print('%s of %s => %s' % (ct, sz, f))
      res = self.project.match(f, self.amnt_results)
      self.results.append(res)

  def evaluate(self, clique_map_csv, output_file=None):
    '''
    clique_map_csv must be a csv containing the pair (work_id, filename) separated by a tab, whereas
    work_id is a string representing the work that groups different covers/versions of a same song.
    filename is a string representing the full path of a given cover song that belongs to the {work_id} group
    '''
    # import code; code.interact(local=dict(globals(), **locals()))
    self.clique_map = self.read_clique_map(clique_map_csv)
    self.calculate_distance_matrix()


    v0 = self.total_covers_in_top_k(self.amnt_results)
    v1 = self.mean_number_of_covers_in_top_k(self.amnt_results)
    v2 = self.mean_rank_of_first_correct_cover()
    v3 = self.mean_avg_precisions()

    if output_file is not None:
      with open(output_file, 'w') as f:
        f.write(self.mirex_output())
    #import code; code.interact(local=dict(globals(), **locals()))

    obj = {
      'Total covers in top %s' % self.amnt_results: v0,
      'Mean number of covers in top %s' % self.amnt_results: v1,
      'Mean rank of first correct cover (MRR)': v2,
      'Mean Average Precision (MAP)': v3,
      'Total cliques': len(set(self.clique_map.values())),
      'Total candidates': len(self.db_filenames),
      'Total queries': len(self.results)
    }
    df1 = pd.DataFrame([obj])

    obj2 = dict([[x+1, self.total_correct_covers_at_rank_pos(x)] for x in range(self.amnt_results)])
    df2 = pd.DataFrame([obj2])
    return df1, df2






  def calculate_distance_matrix(self):
    m = np.zeros((len(self.results), len(self.db_filenames))) + 1
    for i in range(len(self.results)):
      audio, matches = self.results[i]
      query_idx = self.db_filenames.index(audio.filename)
      for match in matches:
        idx2 = self.db_filenames.index(match.filename)
        m[i, idx2] = (1 - match.score)
    self.distance_matrix = m
    return self.distance_matrix, self.queries

  def mean_avg_precisions(self):
    ap_arr = []
    for query_idx in range(len(self.queries)):
      indexes = np.argsort(self.distance_matrix[query_idx])
      ranking = [self.clique_map[self.db_filenames[i]]==self.clique_map[self.queries[query_idx]] for i in indexes]
      ap = util.average_precision(np.array(ranking).astype(int))
      ap_arr.append(ap)
    return np.sum(ap_arr)/len(ap_arr)


  def expected_matrix(self):
    matrix_size = len(self.db_filenames)
    m = np.zeros((matrix_size, matrix_size))
    for idx1 in range(len(self.db_filenames)):
      for idx2 in range(len(self.db_filenames)):
        if self.clique_map[self.db_filenames[idx1]]==self.clique_map[self.db_filenames[idx2]]:
          m[idx1, idx2] = m[idx2, idx1] = 1
    return m

  def mirex_output(self, header_content='noname'):
    tx = '%s\n' % header_content
    for record_idx in range(len(self.db_filenames)):
      tx += '%s\t%s\n' % (record_idx+1, self.db_filenames[record_idx])
    tx += 'Q/R\t'
    tx += '\t'.join([str(record_idx+1) for record_idx in range(len(self.db_filenames))])
    tx += '\n'
    arr = []
    for query_idx in range(len(self.queries)):
      query = self.queries[query_idx]
      res = '\t'.join(["{:.5f}".format(x) for x in self.distance_matrix[query_idx]])
      arr.append('%s\t%s' % (query_idx+1, res))
    tx += '\n'.join(arr)
    return tx

  def total_covers_in_top_k(self, k):
    totals = 0
    for audio, matches in self.results:
      totals += sum([self.clique_map[m.filename]==self.clique_map[audio.filename] for m in matches])
    return totals

  def total_correct_covers_at_rank_pos(self, pos):
    arr = []
    for query_idx in range(len(self.queries)):
      indexes = np.argsort(self.distance_matrix[query_idx])
      ranking = [self.clique_map[self.db_filenames[i]]==self.clique_map[self.queries[query_idx]] for i in indexes]
      arr.append(ranking)
    try:
      return np.array(arr).astype(int).T[pos].sum()
    except:
      return 0

  def mean_number_of_covers_in_top_k(self, k):
    r = [self.total_correct_covers_at_rank_pos(i) for i in range(k-1)]
    return np.sum(r)/len(self.queries)

  def mean_rank_of_first_correct_cover(self):
    rank_arr = []
    for query_idx in range(len(self.queries)):
      indexes = np.argsort(self.distance_matrix[query_idx])
      ranking = [self.clique_map[self.db_filenames[i]]==self.clique_map[self.queries[query_idx]] for i in indexes]
      pos_first_correct_cover = np.argmax(ranking)+1
      rank_arr.append(pos_first_correct_cover)
    return np.sum(rank_arr)/len(rank_arr)


  def matches_map(self):
    arr = []
    for audio, matches in self.results:
      arr.append([int(self.clique_map[m.filename]==self.clique_map[audio.filename]) for m in matches])
    return arr

  def get_filenames(self, filelist):
    fl = open(filelist)
    all_files = fl.read().split('\n')
    fl.close()
    return list(np.sort(list(set(filter(lambda f: os.path.isfile(f), all_files)))))

  def read_clique_map(self, filename):
    f = open(filename, 'r', encoding='utf-8')
    s = list(csv.reader(f, delimiter='\t'))
    f.close()
    return dict([[x[1], x[0]] for x in s])

  def __read_list(self, list_path):
    with open(list_path, 'r') as f:
      x = f.read().splitlines()
    return x

  def query_offset(self, filename, duration):
    with audioread.audio_open(filename) as f:
      dur = f.duration
    return random.random() * (dur - duration)

  def get_query_filename(self, queries_path):
    unique_filename = str(uuid.uuid4().hex) + '.wav'
    return os.path.join(queries_path, unique_filename)


