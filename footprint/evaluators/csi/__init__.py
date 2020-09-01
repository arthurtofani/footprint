#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
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
from . import measures

class CSI:
  records_list_path = None
  db_files = None
  query_files = None
  project = None
  match_results = None
  ranking_size = None
  distance_matrix = None

  def __init__(self, project):
    self.project = project
    self.distance_matrix = None

  def preprocess_batch(self, records_list, max_processors):
    for total_files in self.preprocess(records_list):
      with Pool(max_processors) as pool:
        pool.map(preprocess_audio , total_files)

  def preprocess(self, records_list):
    yield self.get_filenames(records_list)

  def build(self, records_list_path, exclude_path=None, max_processors=None):
    '''
    Receives the list of records that will be
    sent to the database.
    '''
    self.records_list_path = records_list_path
    exclude_files = []
    if exclude_path is not None:
      exclude_files  = self.get_filenames(exclude_path)
    records_files = self.get_filenames(records_list_path)
    self.db_files = [x for x in records_files if x not in exclude_files]
    ct = 0
    total = len(self.db_files)

    if max_processors==None:
      for filename in self.db_files:
        ct+=1
        print('[%s/%s] - Adding %s' % (ct, total, filename))
        self.project.add(filename)
    else:
      with Pool(max_processors) as pool:
        pool.map(self.project.add , self.db_files)


  def match(self, query_files, amnt_results_per_query=10):
    '''
    Performs match between the songs in the query_files file and those ones
    received by the #build method
    '''
    ct = 0
    self.query_files = query_files
    self.match_results = []
    self.amnt_results_per_query = amnt_results_per_query
    self.queries = self.get_filenames(self.query_files)

    sz = len(self.queries)
    for f in self.queries:
      ct+=1
      print('%s of %s => %s' % (ct, sz, f))
      res = self.project.match(f, self.amnt_results_per_query)
      #yield res
      res[0].features = dict()
      self.match_results.append(res)

  def evaluate(self):
    return self.__calculate_distance_matrix()

  def results(self, clique_map, ranking_size=10):
    csi_measures = measures.Measures(self.distance_matrix, clique_map, self.queries, self.db_files)
    v0 = csi_measures.total_covers_in_top_k(ranking_size)
    v1 = csi_measures.mean_number_of_covers_in_top_k(1)
    v1b = csi_measures.mean_number_of_covers_in_top_k(10)
    v2 = csi_measures.mean_rank_of_first_correct_cover()
    v3 = csi_measures.mean_avg_precisions()
    #v4 = csi_measures.mean_reciprocal_rank(ranking_size)

    obj = {
      'Total covers in top %s' % ranking_size: v0,
      'Mean number of covers in top 1': v1,
      'Mean number of covers in top 10': v1b,
      'Mean rank of first correct cover (MR)': v2,
      #'Mean reciprocal rank (MRR)': v4,
      'Mean Average Precision (MAP)': v3,
      'Total cliques': len(set(clique_map.values())),
      'Total candidates': len(self.db_files),
      'Total queries': len(self.match_results)
    }
    df1 = pd.DataFrame([obj])
    obj2 = dict([[x+1, csi_measures.total_correct_covers_at_rank_pos(x)] for x in range(ranking_size)])
    df2 = pd.DataFrame([obj2])
    return df1, df2

  def __calculate_distance_matrix(self):
    m = np.zeros((len(self.match_results), len(self.db_files))) + 1
    for i in range(len(self.match_results)):
      audio, matches = self.match_results[i]
      query_idx = self.queries.index(audio.filename)
      #if i < len(self.db_files):
        #m[i, i] = 0
      for match in matches:
        idx2 = self.db_files.index(match.filename)
        m[i, idx2] = (1 - match.score)
    self.distance_matrix = m
    print('--- distance_matrix len', len(self.distance_matrix))
    return self.distance_matrix, self.queries

  def get_filenames(self, filelist):
    if type(filelist) is np.ndarray:
      all_files = list(filelist)
    else:
      fl = open(filelist)
      all_files = fl.read().split('\n')
      fl.close()
    return list(np.sort(list(set(filter(lambda f: os.path.isfile(f), all_files)))))

#  def __read_list(self, list_path):
#    with open(list_path, 'r') as f:
#      x = f.read().splitlines()
#    return x
