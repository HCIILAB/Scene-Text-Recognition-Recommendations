from __future__ import absolute_import
import sys
import einops
import torch
from torch import nn
import torch.nn.functional as F
from . import create
from torch.nn import CTCLoss
from config import get_args
global_args = get_args(sys.argv[1:])


class ModelBuilder_CTC(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, RGB=False, get_mask=False):
    super(ModelBuilder_CTC, self).__init__()

    input_dim = 3 if RGB else 1
    strides = [(1,1),(2,2),(2,2),(2,1),(2,1),(2,1)]
    self.backbone = create(arch,strides,in_channel=input_dim, get_mask=get_mask)
    self.rnn = nn.LSTM(512,256,num_layers=2,bidirectional=True,batch_first=True)

    self.decoder = nn.Linear(512, rec_num_classes)
    self.rec_crit = CTCLoss(zero_infinity=True)
    self.get_mask = get_mask

  def forward(self, input_dict):
    return_dict = {}
    return_dict['loss'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    encoder_feats = self.backbone(x).squeeze()
    encoder_feats = encoder_feats.transpose(2,1)
    
    encoder_feats = encoder_feats.contiguous()
    encoder_feats,_ = self.rnn(encoder_feats)
    encoder_feats = encoder_feats.contiguous()
    rec_pred = self.decoder(encoder_feats)
    # compute ctc loss
    rec_pred = einops.rearrange(rec_pred, 'B T C -> T B C') # required by CTCLoss
    rec_pred_log_softmax = F.log_softmax(rec_pred,dim=2)
    pred_size = torch.IntTensor([rec_pred.shape[0]]*rec_pred.shape[1]) # (timestep) * batchsize
    loss_rec = self.rec_crit(rec_pred_log_softmax, rec_targets, pred_size, rec_lengths)
    return_dict['loss']['loss_rec'] = loss_rec
    if not self.training:
       return_dict['output']['pred_rec'] = einops.rearrange(rec_pred, 'T B C -> B T C')
    return return_dict

  def inference(self, x):
    encoder_feats = self.backbone(x).squeeze(dim=-2)
    encoder_feats = encoder_feats.transpose(2,1)
    
    encoder_feats = encoder_feats.contiguous()
    encoder_feats,_ = self.rnn(encoder_feats)
    encoder_feats = encoder_feats.contiguous()
    rec_pred = self.decoder(encoder_feats)
    return rec_pred

if __name__=="__main__":
  model = ModelBuilder_CTC('ResNet_Scene',37)
  x = torch.randn(2,3,64,256)
  labels = torch.zeros(2,25).long()
  length = torch.Tensor([7,8]).long()

  input_dict = {}
  input_dict['images'] = x
  input_dict['rec_targets'] = labels
  input_dict['rec_lengths'] = length

  print(model(input_dict).shape)