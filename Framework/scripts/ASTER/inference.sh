CUDA_VISIBLE_DEVICES=0,1 python inference.py \
  --image_path test_img.JPG \
  --height 32 \
  --width 100 \
  --arch ResNet45 \
  --decode_type Attention \
  --with_lstm \
  --max_len 25 \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  --alphabets lowercase \
  --resume weights/ASTER/model_best_acc_bias.pth.tar \