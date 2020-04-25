import unittest
import logging
from footprint.models import Audio
from footprint.models import Project
import footprint.clients as db
import numpy as np
import os
import csv
import random
import footprint.tokenizers as tokenizers
import footprint.evaluators as evaluators
from footprint.features import ocmi
import librosa

class TestEvalCSI(unittest.TestCase):
  '''
  Tests fingerprint evaluator with dummy database
  '''

  def test_smoke(self):
    entries_path = abs_path('fixtures/csi_ytc_entries.txt')
    #expect_path = abs_path('fixtures/csi_ytc_cliques.csv')
    queries_path = abs_path('fixtures/csi_ytc_queries.txt')

    #entries_path = abs_path('fixtures/test/entries_small.txt')
    expect_path = abs_path('fixtures/csi_ytc_cliques.csv')
    #queries_path = abs_path('fixtures/test/queries_small.txt')

    #generate_clique_map(entries_path, expect_path)
    #read_clique_map(filename)

    p = Project(cache=True, cache_folder='/cache')
    p.process_feature('beat_chroma_cens', beat_sync_chroma_cens)
    p.process_feature('chroma_cens', feat_chroma_cens)
    p.process_feature('beat_chroma_ocmi_4', beat_chroma_ocmi_4)
    #p.process_feature('beat_chroma_ocmi_5', beat_chroma_ocmi_5)
    #p.process_feature('beat_chroma_ocmi_3', beat_chroma_ocmi_3)
    #p.process_feature('beat_chroma_docmi_3', beat_chroma_docmi_3)
    #p.process_feature('beat_chroma_docmi_4', beat_chroma_docmi_4)
    #p.process_feature('chroma_ocmi_4', chroma_ocmi_4)
    #p.process_feature('chroma_ocmi_3', chroma_ocmi_3)
    #p.process_feature('bbeat_chroma_ocmi_4', proc_ocmi_feature('beat_chroma_cens', ocmi.ocmi, reshape=(None, 4)))
    #p.process_feature('bbeat_chroma_ocmi_norm_4', proc_ocmi_feature('beat_chroma_cens', ocmi.ocmi_norm, reshape=(None, 4)))
    #p.process_feature('bbeat_chroma_ocmi_norm_4', proc_ocmi_feature('beat_chroma_cens', ocmi.ocmi_norm, reshape=(None, 4)))
    #p.process_feature('bchroma_ocmi_norm_4', proc_ocmi_feature('chroma_cens', ocmi.ocmi_norm, reshape=(None, 4)))

    #p.use_tokenizer('chroma_naive', naive_tokenizer_for_beat_chroma)
    #p.use_tokenizer('chroma_naive2', naive_tokenizer_for_chroma)

    #p.use_tokenizer('ocmi_tokenizer_beat_chroma_ocmi_4', ocmi_tokenizer_beat_chroma_ocmi_4)
    #p.use_tokenizer('ocmi_tokenizer_beat_chroma_ocmi_5', ocmi_tokenizer_beat_chroma_ocmi_5)
    #p.use_tokenizer('ocmi_tokenizer_beat_chroma_ocmi_3', ocmi_tokenizer_beat_chroma_ocmi_3)
    #p.use_tokenizer('ocmi_tokenizer_beat_chroma_docmi_3', ocmi_tokenizer_beat_chroma_docmi_3)
    #p.use_tokenizer('ocmi_tokenizer_beat_chroma_docmi_4', ocmi_tokenizer_beat_chroma_docmi_4)
    #p.use_tokenizer('ocmi_tokenizer_chroma_ocmi_4', ocmi_tokenizer_chroma_ocmi_4)
    #p.use_tokenizer('ocmi_tokenizer_chroma_ocmi_3', ocmi_tokenizer_chroma_ocmi_3)

    #p.use_tokenizer('magic1', magic_tokenizer('bchroma_ocmi_norm_4', min_hash_fns=15, shingle_size=10)) # MAP=0.39
    p.use_tokenizer('magic1', magic_tokenizer('beat_chroma_ocmi_4', min_hash_fns=10, shingle_size=1))

    connect_to_elasticsearch(p)
    p.client.set_scope('csi', 'magic1', 'tokens_by_spaces')
    #p.add('/dataset/YTCdataset/letitbe/test.mp3')
    #import code; code.interact(local=dict(globals(), **locals()))
    evaluator = evaluators.CSI(p)
    print('building')
    evaluator.build(entries_path, exclude_path=queries_path)
    print('matching')
    evaluator.match(queries_path)


    print("\n\n\n==== Results ===")
    df1, df2 = evaluator.evaluate(expect_path)
    print(df1.sum())

    print("==== Total correct covers at rank positions ===")
    print(df2.sum())

    #r = compare_results(result, expect_path)
    self.assertEqual(1, 1)


def generate_clique_map(entries_path, filename):
  f = open(entries_path, 'r', encoding='utf-8')
  files = f.read().split("\n")[0:-1]
  with open(filename, 'w') as f2:
    writer = csv.writer(f2, delimiter='\t')
    for file in files:
      writer.writerow([os.path.dirname(file), file])
  f.close()


def read_clique_map(filename):
  f = open(filename, 'r', encoding='utf-8')
  return list(csv.reader(f, delimiter='\t'))


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

def magic_tokenizer(feature_name, min_hash_fns=10, shingle_size=3):
  def tokenize_method(audio):
    feature = audio.features[feature_name]
    return tokenizers.magic_hash(feature, min_hash_fns=min_hash_fns, shingle_size=shingle_size)
  return tokenize_method

def proc_ocmi_feature(feature_name, method, reshape=(None, None)):
  def feat_method(audio):
    feature = audio.features[feature_name]
    f2 = method(feature)
    return tokenizers.reshape(f2, reshape[0], reshape[1])
  return feat_method



def naive_tokenizer_for_beat_chroma(audio):
  return tokenizers.naive_tokenizer(audio.features['beat_chroma_cens'], pace=30)

def naive_tokenizer_for_chroma(audio):
  return tokenizers.naive_tokenizer(audio.features['chroma_cens'], pace=30)

def ocmi_tokenizer_chroma_ocmi_3(audio):
  chroma = audio.features['beat_chroma_cens']
  return tokenizers.naive_tokenizer(audio.features['chroma_ocmi_3'], pace=30)

def ocmi_tokenizer_chroma_ocmi_4(audio):
  chroma = audio.features['beat_chroma_cens']
  return tokenizers.naive_tokenizer(audio.features['chroma_ocmi_4'], pace=30)

def ocmi_tokenizer_beat_chroma_ocmi_4(audio):
  #return tokenizers.naive_tokenizer(audio.features['beat_chroma_ocmi_4'], pace=30)
  return ocmi.tokenize(audio.features['beat_chroma_ocmi_4'])

def ocmi_tokenizer_beat_chroma_ocmi_3(audio):
  return tokenizers.naive_tokenizer(audio.features['beat_chroma_ocmi_3'], pace=30)

def ocmi_tokenizer_beat_chroma_ocmi_5(audio):
  return tokenizers.naive_tokenizer(audio.features['beat_chroma_ocmi_5'], pace=30)

def ocmi_tokenizer_beat_chroma_docmi_3(audio):
  return tokenizers.naive_tokenizer(audio.features['beat_chroma_docmi_3'], pace=30)

def ocmi_tokenizer_beat_chroma_docmi_4(audio):
  return tokenizers.naive_tokenizer(audio.features['beat_chroma_docmi_4'], pace=30)

def feat_chroma_cens(audio):
  print('running chroma for ', audio.filename)
  return librosa.feature.chroma_cens(audio.y, audio.sr)

def chroma_ocmi_3(audio):
  chroma = audio.features['chroma_cens']
  return ocmi.calc_ocmi(chroma, 3)

def chroma_ocmi_4(audio):
  chroma = audio.features['chroma_cens']
  return ocmi.calc_ocmi(chroma, 4)

def beat_chroma_ocmi_4(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.calc_ocmi(chroma, 4)

def beat_chroma_ocmi_3(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.calc_ocmi(chroma, 3)

def beat_chroma_ocmi_5(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.calc_ocmi(chroma, 5)

def beat_chroma_docmi_3(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.calc_docmi(chroma, 3)

def beat_chroma_docmi_4(audio):
  chroma = audio.features['beat_chroma_cens']
  return ocmi.calc_docmi(chroma, 4)


def beat_sync_chroma_cens(audio):
  chroma =  librosa.feature.chroma_cens(audio.y, sr=audio.sr, hop_length=2**10)
  audio.load_beats()
  beat_f = librosa.util.fix_frames(audio.beats, x_max=chroma.shape[1])
  return librosa.util.sync(chroma, beat_f, aggregate=np.median)

def connect_to_elasticsearch(p):
  cli = db.elasticsearch.Connection(host='elasticsearch', port=9200)
  cli.clear_index('csi')
  cli.setup_index('csi', initial_settings())
  p.set_connection(cli)


def initial_settings():
  return {
    "settings" : {
      "analysis" : {
        "analyzer" : {
          "tokens_by_spaces": {
            "tokenizer": "divide_tokens_by_spaces"
          }
        },
        "tokenizer": {
          "divide_tokens_by_spaces": {
            "type": "simple_pattern_split",
            "pattern": " "
          }
        }
      }
    }
  }

if __name__ == '__main__':
  unittest.main()

#      import code; code.interact(local=dict(globals(), **locals()))
# python3 -m unittest test.eval_csi.TestEvalCSI.test_smoke
