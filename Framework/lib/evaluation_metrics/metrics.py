from __future__ import absolute_import

import numpy as np
import editdistance
import string
import math

import torch
import torch.nn.functional as F
from ..utils import to_numpy


def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()
  #return text

def get_str_list(output,target, converter, beam_width=0):

  output = F.softmax(output,dim=2) # B T C -> B T C
  score, predicted = output.max(2) # B T C -> B T
  pred_list,eos_idx = converter.decode(predicted)
  score_list = []
  for i in range(score.shape[0]):
    score_list.append(torch.prod(score[i][:eos_idx[i]]).item())
  pred_list = [_normalize_text(pred) for pred in pred_list]
  targ_list = [_normalize_text(targ) for targ in target]
  return pred_list, targ_list, score_list

def Accuracy(output, target, converter):
  pred_list, target_list, score_list = get_str_list(output,target, converter)
  acc_list = [(pred == targ) for pred, targ in zip(pred_list, target_list)]
  accuracy = 1.0 * sum(acc_list) / len(acc_list)
  return pred_list, score_list, accuracy

def RecPostProcess(output, target, score, dataset=None):
  pred_list, targ_list = get_str_list(output, target, dataset) #id转char
  max_len_labels = output.size(1)
  score_list = []

  score = to_numpy(score)
  for i, pred in enumerate(pred_list):
    len_pred = len(pred) + 1 # eos should be included
    len_pred = min(max_len_labels, len_pred) # maybe the predicted string don't include a eos.
    score_i = score[i,:len_pred]
    score_i = math.exp(sum(map(math.log, score_i)))
    score_list.append(score_i)

  return pred_list, targ_list, score_list