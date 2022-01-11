from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import os
import math
import argparse

parser = argparse.ArgumentParser(description="ASTER")
# data
parser.add_argument('--synthetic_train_data_dir', nargs='+', type=str, metavar='PATH')
parser.add_argument('--test_data_dir', nargs='+', type=str, metavar='PATH')
parser.add_argument('--alphabets',type=str,choices=['lowercase','allcases','allcases_symbols'])
parser.add_argument('--punc',action='store_true',help="add  ,.!?;': to the alphabets")
parser.add_argument('--height',type=int,default=192, help='height of the input image')
parser.add_argument('--width',type=int,default=2048, help='width of the input image')
parser.add_argument('--padresize',action='store_true',help='to use pad resize (recommend on iam , not on scene text)')
parser.add_argument('--keepratioresize',action='store_true',help='to use pad resize (recommend on iam , not on scene text)')
parser.add_argument('--RGB',action='store_true',help='to use RGB image')

parser.add_argument('--lower',action='store_true',help='lower all labels')
parser.add_argument('--max_len',type=int,default=128, help='max decode and encode length')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--keep_ratio', action='store_true', help='length fixed or lenghth variable.')
parser.add_argument('--lexicon_type', type=str, default='0', choices=['0', '50', '1k', 'full'], help='which lexicon associated to image is used.')
parser.add_argument('--image_path', type=str, default='', help='the path of single image, used in demo.py.')
# model
parser.add_argument('--arch', type=str, default='ResNet45_MIN')
parser.add_argument('--decode_type', type=str,choices=['Attention','CTC', 'DAN', 'ACE','SAR'])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_group', type=int, default=1)
# for attention
parser.add_argument('--tps_inputsize', nargs='+', type=int, default=[32, 64])
parser.add_argument('--tps_outputsize', nargs='+', type=int, default=[32, 100])
parser.add_argument('--STN_ON', action='store_true',help='add the stn head.')
parser.add_argument('--tps_margins', nargs='+', type=float, default=[0.05,0.05])
parser.add_argument('--stn_activation', type=str, default='none')
parser.add_argument('--num_control_points', type=int, default=20)
parser.add_argument('--stn_with_dropout', action='store_true', default=False)
parser.add_argument('--decoder_sdim', type=int, default=512,help="the dim of hidden layer in decoder.")
parser.add_argument('--attDim', type=int, default=512, help="the dim for attention.")

## lstm
parser.add_argument('--with_lstm', action='store_true', help='whether append lstm after cnn in the encoder part.')
# optimizer
parser.add_argument('--lr', type=float, default=3e-4,help="learning rate")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])
# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--SGD',action='store_true',help='whether to use adamdelta(recommened on scene text)')
parser.add_argument('--adamdelta',action='store_true',help='whether to use adamdelta(recommened on scene text)')
parser.add_argument('--randomsequentialsampler',action='store_true',help='in my case, i use random sequential sampler in scene text')
parser.add_argument('--augmentation', action='store_true', help='using data augmentation')
parser.add_argument('--warmup',type=int,default=-1, help='warm up')

parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--epochs', type=int, default= 400)
parser.add_argument('--stepLR',type=int, nargs='+', help='step learning rate, ex [100,200]')
parser.add_argument('--iter_mode', action='store_true', help="train on epoch model(small dataset) or on iteration model(large dataset)")
parser.add_argument('--start_save', type=int, default=0,help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', default=True, type=bool,help='whether use cuda support.')
parser.add_argument('--tensorboard_freq_iter',type=int, default=10,help='frequency to log tensorbaord')
parser.add_argument('--evaluation_freq_iter',type=int, default=1000,help='frequency to log tensorbaord')
parser.add_argument('--tensorboard_freq_epoch',type=int, default=1,help='frequency to log tensorbaord')
parser.add_argument('--evaluation_freq_epoch',type=int, default=10,help='frequency to log tensorbaord')

# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy')
parser.add_argument('--evaluate_with_lexicon', action='store_true', default=False)
parser.add_argument('--beam_width', type=int, default=1) # something wrong with beam search, use grady search as defaults
# logs
parser.add_argument('--logs_dir',type=str)

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args