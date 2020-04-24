from . import base
import hashedindex
from collections import Counter
from footprint.models.match_result import MatchResult

class Connection(base.DbConnection):
  index = hashedindex.HashedIndex()

  def query(self, audio, amnt_results=10):
    results = self.__perform_match(audio, amnt_results)
    return audio, results

  def add(self, audio):
    for word in self.__get_tokens(audio):
      self.index.add_term_occurrence(word, audio.filename)
    return audio

  def __perform_match(self, audio, k):
    counter = Counter()
    for word in self.__get_tokens(audio):
      try:
        counter = counter + self.index.get_documents(word)
      except IndexError:
        pass
    return self.__generate_results(counter.most_common(k))

  def __get_tokens(self, audio):
    '''
    For this dummy db only one set of tokens
    is used - the first in token's dictionary.
    '''
    key =  list(audio.tokens.keys())[0]
    return audio.tokens[key].split(' ')

  def __generate_results(self, top_k_records):
    if len(top_k_records)==0:
      return []
    score_base = top_k_records[0][1]
    #return counter
    return [MatchResult(r[0], r[1]/score_base, []) for r in top_k_records]

