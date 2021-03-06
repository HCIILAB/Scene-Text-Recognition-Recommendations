CUDA_VISIBLE_DEVICES=0,1 python inference.py \
  --image_path test_img.JPG \
  --height 32 \
  --width 100 \
  --arch ResNet45 \
  --decode_type CTC \
  --with_lstm \
  --max_len 25 \
  --alphabets lowercase \
  --lower \
  --iter_mode \
  --evaluate \
  --resume weights/CRNN/model_best_acc_bias.pth.tar \