from __future__ import absolute_import

from .metrics import Accuracy


__factory = {
  'accuracy': Accuracy,
}

def names():
  return sorted(__factory.keys())

def factory():
  return __factory