from __future__ import absolute_import
import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO         # Python 3.x

from .osutils import mkdir_if_missing

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      mkdir_if_missing(os.path.dirname(fpath))
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
      self.file.flush()
      os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
      self.file.close()


class TFLogger(object):
  def __init__(self, log_dir=None):
    """Create a summary writer logging to log_dir."""
    if log_dir is not None:
      mkdir_if_missing(log_dir)
    self.writer = SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    self.writer.add_scalar(tag,value,step)

  def close(self):
    self.writer.close()