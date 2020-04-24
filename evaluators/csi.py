import numpy as np
import glob
import random
from collections import defaultdict
import librosa
import pandas as pd
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
    for f in self.get_filenames(query_list_path):
      res = self.project.match(f, self.amnt_results)
      self.results.append(res)

  def evaluate(self, clique_map_csv):
    '''
    clique_map_csv must be a csv containing the pair (work_id, filename) separated by a tab, whereas
    work_id is a string representing the work that groups different covers/versions of a same song.
    filename is a string representing the full path of a given cover song that belongs to the {work_id} group
    '''
    # import code; code.interact(local=dict(globals(), **locals()))
    self.clique_map = self.read_clique_map(clique_map_csv)
    self.calculate_correlation_matrix()

    v0 = self.total_covers_in_top_k(self.amnt_results)
    v1 = self.mean_number_of_covers_in_top_k(self.amnt_results)
    v2 = self.mean_rank_of_first_correct_cover()
    v3 = self.mean_avg_precisions()

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

    obj2 = dict([[x+1, self.total_correct_covers_at_rank_pos(x+1)] for x in range(self.amnt_results)])
    df2 = pd.DataFrame([obj2])
    return df1, df2






  def calculate_correlation_matrix(self):
    arr = []
    for audio, matches in self.results:
      results = defaultdict()
      results['query'] = audio.filename
      d = dict([[m.filename, m.score] for m in matches])
      for candidate in self.db_filenames:
        try:
          results[candidate] = d[candidate]
        except KeyError:
          results[candidate] = 0
      arr.append(results)

    self.correlation_matrix = pd.DataFrame(arr)
    return self.correlation_matrix

  def total_covers_in_top_k(self, k):
    totals = 0
    for audio, matches in self.results:
      totals += sum([self.clique_map[m.filename]==self.clique_map[audio.filename] for m in matches])
    return totals

  def total_correct_covers_at_rank_pos(self, pos):
    if self.amnt_results < pos:
      raise 'Max positions available: %s' % self.amnt_results
    totals = 0
    for audio, matches in self.results:
      try:
        idx = np.argsort([m.score for m in matches])[pos-1]
        if self.clique_map[matches[idx].filename]==self.clique_map[audio.filename]:
          totals+=1
      except IndexError:
        pass
    return totals

  def mean_number_of_covers_in_top_k(self, k):
    arr = []
    for audio, matches in self.results:
      vl = sum([self.clique_map[m.filename]==self.clique_map[audio.filename] for m in matches])
      arr.append(vl)
    return np.average(arr)

  def mean_avg_precisions(self):
    return util.mean_average_precision(self.matches_map())

  def mean_rank_of_first_correct_cover(self):
    # aka Mean Reciprocal Rank
    # https://gist.github.com/bwhite/3726239 s
    arr = []
    for audio, matches in self.results:
      rnk = [int(self.clique_map[m.filename]==self.clique_map[audio.filename]) for m in matches]
      arr.append(rnk)
    rs = [np.asarray(r).nonzero()[0] for r in arr]
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

  def matches_map(self):
    arr = []
    for audio, matches in self.results:
      arr.append([int(self.clique_map[m.filename]==self.clique_map[audio.filename]) for m in matches])
    return arr

  def get_filenames(self, filelist):
    fl = open(filelist)
    all_files = fl.read().split('\n')
    fl.close()
    return list(set(filter(lambda f: os.path.isfile(f), all_files)))

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


