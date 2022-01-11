from __future__ import absolute_import
import os
import errno


def mkdir_if_missing(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def make_symlink_if_not_exists(real_path, link_path):
  '''
  param real_path: str the path linked
  param link_path: str the path with only the symbol
  '''
  if not os.path.exists(real_path):
    os.makedirs(real_path)

  cmd = 'ln -s {0} {1}'.format(real_path, link_path)
  os.system(cmd)