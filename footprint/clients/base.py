class DbConnection:

  config = None
  def __init__(self, **kwargs):
    self.config = kwargs
    self.on_init()
    return

  def on_init():
    pass

  def query(self, audio, amnt_results=10):
    raise NotImplementedError

  def add(self, audio):
    raise NotImplementedError

  def clear(self):
    raise NotImplementedError

  def setup(self):
    raise NotImplementedError
