from . import base
import warnings
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from footprint.models.match_result import MatchResult

class Connection(base.DbConnection):

  es = None
  current_index = None
  current_analyzer = None
  current_tokens_key = None

  def on_init(self):
    self.es = Elasticsearch(
            [self.config.get('host') or 'localhost'],
            scheme=(self.config.get('protocol') or 'http'),
            port=(self.config.get('port') or 9200),
    )

  def setup_index(self, index_name, settings):
    self.es.indices.create(index=index_name, body=settings, ignore=400)

  def clear_index(self, index_name):
    try:
      self.es.indices.delete(index=index_name)
    except NotFoundError:
      print('index %s not found, skipping' % index_name)

  def set_scope(self, index_name, tokens_key, analyzer):
    self.current_index = index_name
    self.current_analyzer = analyzer
    self.current_tokens_key = tokens_key

  def splitter(self, n, s):
    pieces = s.split()
    return [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]

  def query(self, audio, amnt_results=10, split_length=500):
    results = []
    matches = []
    tokens = audio.tokens[self.current_tokens_key]
    try:
      for piece in self.splitter(split_length, tokens):
        match_obj = { 'match':{ self.current_tokens_key: { 'query': piece } } }
        matches.append(match_obj)
      body = {'query': {
                'bool': {
                  'should': matches
                  }
                },
              'from' : 0, 'size' : amnt_results,
              }
    except AttributeError:
      warnings.warn("empty tokens: %s / %s" % (audio.filename, self.current_tokens_key))
      return (audio, [])
    res = self.es.search(index=self.current_index, doc_type='tokens', body=body)
    results = []
    for idx in range(len(res['hits']['hits'])):
      filename = res['hits']['hits'][idx]['_id']
      score = res['hits']['hits'][idx]['_score']/res['hits']['max_score']
      results.append(MatchResult(filename, score, []))
    return audio, results

  def add(self, audio):
    self.es.index(index=self.current_index, doc_type='tokens', id=audio.filename, body=dict(audio.tokens))
    return audio

  def __create_document(self, doc):
    self.es.create()

  def __search_document(self, doc, analyzer):
    self.es.search(doc=doc, analyzer=analyzer)


doc = {
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

index_name = 'test'
shingle = {
  "settings": {
    "analysis": {
      "analyzer": {
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
  },
  "mappings":{
      index_name: {
         "properties":{
            "content":{
               "search_analyzer":"analyzer_shingle",
               "index_analyzer":"analyzer_shingle",
               "type":"string"
            }
         }
      }
   }
}

search_query = {
  "query": {
    "bool" : {
      "must" : {
        "term" : { "user" : "kimchy" }
      },
      "filter": {
        "term" : { "tag" : "tech" }
      },
      "must_not" : {
        "range" : {
          "age" : { "gte" : 10, "lte" : 20 }
        }
      },
      "should" : [
        { "term" : { "tag" : "wow" } },
        { "term" : { "tag" : "elasticsearch" } }
      ],
      "minimum_should_match" : 1,
      "boost" : 1.0
    }
  }
}
{
  "query": {
    "match" : {
      "content" : {
          "query" : "this is a test"
      }
    }
  }
}
