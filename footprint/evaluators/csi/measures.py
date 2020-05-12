import numpy as np

class Measures:
  distance_matrix = None
  clique_map = None
  queries = None
  db_records = None

  def __init__(self, distance_matrix, clique_map, queries, db_records):
    self.distance_matrix = distance_matrix
    self.clique_map = clique_map
    self.queries = queries
    self.db_records = db_records

  def mean_avg_precisions(self):
    ap_arr = []
    for ranking, idx in self.__compute_ranking():
      ap = average_precision(np.array(ranking).astype(int))
      ap_arr.append(ap)
    return np.sum(ap_arr)/len(ap_arr)

  def mean_rank_of_first_correct_cover(self):
    rank_arr = []
    for ranking, idx in self.__compute_ranking():
      pos_first_correct_cover = np.argmax(ranking)+1
      rank_arr.append(pos_first_correct_cover)
    return np.sum(rank_arr)/len(rank_arr)

  def total_covers_in_top_k(self, k):
    return sum([self.total_correct_covers_at_rank_pos(x) for x in range(k)])

  def mean_number_of_covers_in_top_k(self, k):
    #r = [self.total_correct_covers_at_rank_pos(i, self.distance_matrix, self.clique_map, self.queries, self.db_records) for i in range(k-1)]
    return self.total_covers_in_top_k(k)/len(self.queries)

  def total_correct_covers_at_rank_pos(self, pos):
    arr = []
    for ranking, idx in self.__compute_ranking():
      arr.append(ranking)
    try:
      return np.array(arr).astype(int).T[pos].sum()
    except:
      return 0

  def __compute_ranking(self):
    amnt_queries = len(self.distance_matrix)
    for query_idx in range(amnt_queries):
      indexes = np.argsort(self.distance_matrix[query_idx])
      ranking = [self.clique_map[self.db_records[i]]==self.clique_map[self.queries[query_idx]] for i in indexes]
      yield ranking, query_idx

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)
