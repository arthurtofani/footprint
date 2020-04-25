import unittest
import logging
from footprint.models import Audio
from footprint.models import Project
import footprint.clients as db
import os
import random
import footprint.tokenizers as tokenizers
import footprint.evaluators as evaluators
import librosa

class TestEvalFingerprintDummy(unittest.TestCase):
  '''
  Tests fingerprint evaluator with dummy database
  '''

  def test_smoke(self):
    random.seed(30)
    filename = '/dataset/YTCdataset/letitbe/test.mp3'
    bucket_name = 'test'
    p = load_project(cache=True)
    set_dummy_connection(p)
    evaluator = evaluators.Fingerprint(p)
    entries = abs_path('fixtures/fgpt_entries.txt')
    #queries_txt = abs_path('fixtures/queries.txt')
    queries_path = abs_path('/cache/queries')
    evaluator.build(entries)
    queries_list_path = '/cache/queries.txt'
    expect_path = '/cache/expect.txt'
    evaluator.generate_queries(entries, 5, duration=40, queries_path=queries_path, queries_list_path=queries_list_path, results=expect_path)
    evaluator.match(queries_list_path)
    result = evaluator.result()
    r = compare_results(result, expect_path)
    self.assertEqual(r, 1)
    # import code; code.interact(local=dict(globals(), **locals()))

def compare_results(result, expectation_path):
  file = open(expectation_path, 'r')
  fil = file.read()
  expected = dict([x.split('\t') for x in fil.split('\n')])
  file.close()
  comparisons = [expected[query]==found for query, found in result]
  return sum(comparisons)/len(comparisons)

def abs_path(path):
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirname, path)

def naive_tokenizer_for_chroma(audio):
  return tokenizers.naive_tokenizer(audio.features['chroma_cens'], pace=30)

def feat_chroma_cens(audio):
  print('running chroma for ', audio.filename)
  return librosa.feature.chroma_cens(audio.y, audio.sr)

def set_dummy_connection(p):
  cli = db.dummy.Connection(**{'var1': True})
  p.set_connection(cli)

def load_project(cache=True):
  p = Project(cache=cache, cache_folder='/cache')
  p.process_feature('chroma_cens', feat_chroma_cens)
  p.use_tokenizer('chroma_naive', naive_tokenizer_for_chroma)
  return p


if __name__ == '__main__':
  unittest.main()

#      import code; code.interact(local=dict(globals(), **locals()))
# python3 -m unittest test.eval_fingerprint.TestEvalFingerprintDummy.test_smoke
