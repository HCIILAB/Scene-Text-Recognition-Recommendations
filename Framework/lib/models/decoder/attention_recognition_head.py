from __future__ import absolute_import

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

device =torch.device('cuda' if torch.cuda.is_available() else 'gpu')
class AttentionRecognitionHead(nn.Module):
  """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """
  def __init__(self, num_classes, in_planes, sDim, attDim, max_len_labels):
    super(AttentionRecognitionHead, self).__init__()
    self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
    self.in_planes = in_planes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels

    self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim)

  def forward(self, x):
    x, targets, lengths = x
    batch_size = x.size(0)
    # Decoder
    state = torch.zeros(1, batch_size, self.sDim).to(device)
    outputs = []

    for i in range(max(lengths)):
      if i == 0:
        y_prev = torch.zeros((batch_size)).fill_(self.num_classes).to(device) # the last one is used as the <BOS>.
      else:
        y_prev = targets[:,i-1]

      output, state = self.decoder(x, state, y_prev)
      outputs.append(output)
    outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    return outputs

  # inference stage.
  def sample(self, x):
    batch_size = x.size(0)
    # Decoder
    state = torch.zeros(1, batch_size, self.sDim).to(device)
    outputs = []
    for i in range(self.max_len_labels):
      if i == 0:
        y_prev = torch.zeros((batch_size)).fill_(self.num_classes).to(device)
      else:
        y_prev = predicted

      output, state = self.decoder(x, state, y_prev)
      outputs.append(output)
      output = F.softmax(output, dim=1)
      _, predicted = output.max(1)
    outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    return outputs

class AttentionUnit(nn.Module):
  def __init__(self, sDim, xDim, attDim):
    super(AttentionUnit, self).__init__()

    self.sDim = sDim
    self.xDim = xDim
    self.attDim = attDim

    self.sEmbed = nn.Linear(sDim, attDim)
    self.xEmbed = nn.Linear(xDim, attDim)
    self.wEmbed = nn.Linear(attDim, 1)

    # self.init_weights()

  def init_weights(self):
    init.normal_(self.sEmbed.weight, std=0.01)
    init.constant_(self.sEmbed.bias, 0)
    init.normal_(self.xEmbed.weight, std=0.01)
    init.constant_(self.xEmbed.bias, 0)
    init.normal_(self.wEmbed.weight, std=0.01)
    init.constant_(self.wEmbed.bias, 0)

  def forward(self, x, sPrev):
    batch_size, T, _ = x.size()                      # [b x T x xDim]
    x = x.view(-1, self.xDim)                        # [(b x T) x xDim]
    xProj = self.xEmbed(x)                           # [(b x T) x attDim]
    xProj = xProj.view(batch_size, T, -1)            # [b x T x attDim]

    sPrev = sPrev.squeeze(0)
    sProj = self.sEmbed(sPrev)                       # [b x attDim]
    sProj = torch.unsqueeze(sProj, 1)                # [b x 1 x attDim]
    sProj = sProj.expand(batch_size, T, self.attDim) # [b x T x attDim]

    sumTanh = torch.tanh(sProj + xProj)
    sumTanh = sumTanh.view(-1, self.attDim)

    vProj = self.wEmbed(sumTanh) # [(b x T) x 1]
    vProj = vProj.view(batch_size, T)

    alpha = F.softmax(vProj, dim=1) # attention weights for each sample in the minibatch

    return alpha


class DecoderUnit(nn.Module):
  def __init__(self, sDim, xDim, yDim, attDim):
    super(DecoderUnit, self).__init__()
    self.sDim = sDim
    self.xDim = xDim
    self.yDim = yDim
    self.attDim = attDim
    self.emdDim = attDim

    self.attention_unit = AttentionUnit(sDim, xDim, attDim)
    self.tgt_embedding = nn.Embedding(yDim+1, self.emdDim) # the last is used for <BOS> 
    self.gru = nn.GRU(input_size=xDim+self.emdDim, hidden_size=sDim, batch_first=True)
    self.fc = nn.Linear(sDim, yDim)

    # self.init_weights()

  def init_weights(self):
    init.normal_(self.tgt_embedding.weight, std=0.01)
    init.normal_(self.fc.weight, std=0.01)
    init.constant_(self.fc.bias, 0)

  def forward(self, x, sPrev, yPrev):
    # x: feature sequence from the image decoder.
    batch_size, T, _ = x.size()
    alpha = self.attention_unit(x, sPrev)
    context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
    yProj = self.tgt_embedding(yPrev.long())
    # self.gru.flatten_parameters()
    output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), sPrev)
    output = output.squeeze(1)

    output = self.fc(output)
    return output, state