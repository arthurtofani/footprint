import footprint
from footprint.models import Feature
from footprint.clients.elasticsearch import Client
from footprint import evaluators
import librosa
import numpy as np

def feat_chroma_cens(audio):
  return librosa.feature.chroma_cens(audio.y, audio.sr)

def feat_chroma_cqt(audio):
  return librosa.feature.chroma_cqt(audio.y, audio.sr)

def beat_sync_chroma_cens(audio):
  chroma =  librosa.feature.chroma_cens(audio.y, audio.sr)
  beat_f = librosa.util.fix_frames(beats, x_max=chroma.shape[1])
  return librosa.util.sync(chroma, beat_f, aggregate=np.median)

def chroma_cens_vec2str(audio):
  return somelib.tokenize(audio.feature['chroma_cens'])

def beat_chroma_vec2str(audio):
  return somelib.tokenize(audio.feature['beat_chroma'])


client = Client('blablaa')
client.set_analyzer('an1', {'a': 1})
client.setup()

p = Project(cache=True, cache_folder='/cache')
p.set_connection(client)
p.process_feature('chroma_cens', feat_chroma_cens)
p.process_feature('beat_chroma', beat_sync_chroma_cens)
p.use_tokenizer('chr_cens_vec2str', chroma_cens_vec2str)
p.use_tokenizer('beatchroma_vec2str', beat_chroma_vec2str)

#p.add('/dataset/YTCdataset/letitbe/test.mp3')
#p.match('/dataset/YTCdataset/letitbe/test.mp3')


evaluator = evaluators.Fingerprint(p)
evaluator.build('/cache/records_list.txt')

#optional
evaluator.generate_queries(50, src='/cache/records_list.txt', output='/cache/query_list.txt', results='/cache/results.txt')
(query_file_path, db_file_path) = evaluator.match('/cache/query_list.txt')
evaluator.results()

