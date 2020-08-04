from . import base
import warnings
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from footprint.models.match_result import MatchResult

class Connection(base.DbConnection):

  es = None
  current_index = None
  current_analyzer = None
  current_tokens_keys = None

  def on_init(self):
    self.es = Elasticsearch(
            [self.config.get('host') or 'localhost'],
            scheme=(self.config.get('protocol') or 'http'),
            port=(self.config.get('port') or 9200),
            timeout=30
    )

  def setup_index(self, index_name, settings):
    self.es.indices.create(index=index_name, body=settings)

  def clear_index(self, index_name):
    try:
      self.es.indices.delete(index=index_name)
    except NotFoundError:
      print('index %s not found, skipping' % index_name)

  def set_scope(self, index_name, tokens_keys, analyzer, exclude_query_id=True):
    self.current_index = index_name
    self.current_analyzer = analyzer
    self.exclude_query_id = exclude_query_id
    if isinstance(tokens_keys, list):
      self.current_tokens_keys = tokens_keys
    else:
      self.current_tokens_keys = [tokens_keys]

  def splitter(self, n, s):
    pieces = s.split()
    return [" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n)]

  def exclude_query_id_from_search(self, body, document_filename):
    obj = {
      'must_not': {
        'term': {'_id' : document_filename}
      }
    }
    body['query']['bool'].update(obj)

  def create_matches(self, document, split_length):
    matches = []
    for token_key in self.current_tokens_keys:
      tokens = document.tokens[token_key]
      for piece in self.splitter(split_length, tokens):
        match_obj = { 'match':{ token_key: { 'query': piece } } }
        matches.append(match_obj)
    return matches

  def query(self, document, amnt_results=10, split_length=500):
    results = []
    #try:
    body = {'query': {
              'bool': {
                'should': self.create_matches(document, split_length)
              }
            },
            'from' : 0,
            'size' : amnt_results
          }
    if self.exclude_query_id:
      self.exclude_query_id_from_search(body, document.filename)
    #except AttributeError:
      #warnings.warn("empty tokens: %s / %s" % (document.filename, ''))
      #return (document, [])
    res = self.es.search(index=self.current_index, doc_type='tokens', body=body, request_timeout=240)
    results = []
    for idx in range(len(res['hits']['hits'])):
      filename = res['hits']['hits'][idx]['_id']
      try:
        score = res['hits']['hits'][idx]['_score']/res['hits']['max_score']
      except ZeroDivisionError:
        score = 0
      tokens = res['hits']['hits'][idx]['_source']
      del tokens['timeout']
      results.append(MatchResult(filename, score, tokens))
    return document, results

  def add(self, document):
    body = dict(document.tokens).copy()
    body['timeout'] = 30
    self.es.index(index=self.current_index, doc_type='tokens', id=document.filename, body=body)
    return document

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
