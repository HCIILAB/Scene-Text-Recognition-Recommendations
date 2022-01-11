import torch
import PIL.Image as Image
import torchvision
import time
import sys
import torch.nn.functional as F

from config import get_args
from lib.models.model_builder_Attention import ModelBuilder_Att
from lib.models.model_builder_CTC import ModelBuilder_CTC
from lib.utils.labelmaps import CTCLabelConverter, AttentionLabelConverter
from lib.utils.serialization import load_checkpoint
from lib.datasets.dataset import Padresize, resizeNormalize
from lib.utils.alphabets import get_alphabets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

  args = get_args(sys.argv[1:])
  args.alphabets = get_alphabets(args.alphabets)

  """ Set up model with converter """
  if args.decode_type == 'CTC':
    model = ModelBuilder_CTC(arch=args.arch, rec_num_classes=len(args.alphabets)+1) # +1 for [blank]
    converter = CTCLabelConverter(args.alphabets, args.max_len)
  elif args.decode_type == 'Attention':
    model = ModelBuilder_Att(arch=args.arch,rec_num_classes=len(args.alphabets)+2, #+3 for <EOS>, <PAD>
            sDim=args.decoder_sdim, attDim=args.attDim,max_len_labels=args.max_len,STN_ON=args.STN_ON)
    converter = AttentionLabelConverter(args.alphabets,args.max_len)
  checkpoint = load_checkpoint(args.resume)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()

  # creat transform
  if args.padresize:

    transform = Padresize(args.height, args.width)
  else:

    transform = resizeNormalize((args.width, args.height))
  # load img
  img = Image.open(args.image_path).convert('L')
  img = transform(img).unsqueeze(0).to(device)

  # inferrence
  with torch.no_grad():
    time1=time.time()
    pred = model.inference(img)
    if len(pred) == 2:
      torchvision.utils.save_image(pred[1],'rectified_img.jpg')
      pred = pred[0]
  # convert prediction
    pred = F.softmax(pred,dim=2) # B T C -> B T C
    score, predicted = pred.max(2) # B T C -> B T
    pred_string,eos_idx = converter.decode(predicted)
    pred_score = torch.prod(score[:eos_idx[0]]).item()
    
    time_cost = time.time() - time1
    print('Prediction: ',pred_string, 'Predcition Score: ',pred_score, 'Cost time: ',time_cost)