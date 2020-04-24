class MatchResult:
  filename = None
  score = 0
  tokens = None

  def __init__(self, filename, score, tokens):
    self.filename = filename
    self.score = score
    self.tokens = tokens

