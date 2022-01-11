# <center>Scene Text Recognition Framework
<h1 align="center">
    <br>
    <img src="../img/framework.JPG" >
</h1>

## Acknowledgment
- **This framework is based on [ayumiymk/aster.pytorch](https://github.com/ayumiymk/aster.pytorch). Sincerely thanks to ayumiymk for his great contribution to the community of STR**

## Content
- 1. [Algorithms](#1-algorithms)
  - 1.1. [CRNN](#11-crnn)
  - 1.2. [ASTER](#12-aster)
- 2. [Train](#2-train)
  - 2.1. [Prepare Datasets](#21-preparing-datasets)
  - 2.2. [Set up Configurations](#22-set-up-configurations)
  - 2.3. [Start Training](#23-start-training)
  - 2.4. [Check the Training Process](#24-check-the-training-process)
- 3. [Test](#3test)
- 4. [Inference](#4inference)

## 1. Algorithms
### 1.1. CRNN
- **Paper**: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://ieeexplore.ieee.org/abstract/document/7801919)
- **Our Reimplementation:** 
  * **Note**: We use ResNet45 as the backbone of CRNN
- **Checkpoint**: [Checkpoint for CRNN, 提取码;DLVC](https://pan.baidu.com/s/1AUWFXB8fI4uvn1mJZhEvpQ)
  
||IIIT5K|IC03|IC13|IC15|SVT|SVTP|CUTE|Average|
|----|----|----|----|----|----|----|----|----|
|CRNN_Original|78.2%|-|86.7%|-|80.8%|-|-|-|
|CRNN_Reimplemented|91.63%|91.00%|89.75%|75.98%|85.63%|73.95%|78.47%|83.77%|

### 1.2. ASTER
- **Paper**: [ASTER: An Attentional Scene Text Recognizer with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027)
- **Our Reimplementation:**
  * **Note**: We don't use the bidirectional decoder menthioned in the orginal paper
- **Checkpoint**: [Checkpoint for ASTER, 提取码:DLVC](https://pan.baidu.com/s/1ICnUYxOXpDs-Ws3DdNXLfg)

||IIIT5K|IC03|IC13|IC15|SVT|SVTP|CUTE|Average|
|----|----|----|----|----|----|----|----|----|----|
|ASTER_Original|92.67%|-|90.74%|76.1%|91.16%|78.76%|76.39%|
|ASTER_Reimplemented|94.2%|94.00%|94.31%|82.22%|90.26%|83.72%|84.38%|89.013%|

## 2. Train
- With the following steps, you can train CRNN and ASTER on Scene text or your own datasets.
### 2.1. Preparing Datasets
#### Training Set and Evaluation Set
- 1. To train a good Scene text recognizer, you need massive synthetic data or enough real data. We have prepare the `.lmdb` forms of `SynthText` and `MJSynth` dataset for you. Also, the commonly used benchmark are also provided. Download them from here: 
  * [Scene text datasets(提取码:emco)](https://pan.baidu.com/s/1PBJf-BtFa7mLkltIfTXPhQ)
- 2. Unzip file `NIPS2014.zip` and `CVPR2016.zip`. Then you can put the training set and benchmarks together of seperately.

### 2.2 Set up Configurations
- You can simply modify the `.sh` file in [scripts(ASTER)](scripts/ASTER/train.sh).
- Basically, you just need to speicify the training set path `synthetic_train_data_dir` and one benchmark path `test_data_dir` in any one of the `train.sh` file

### 2.3 Start Training
- **Train ASTER**: 
```Bash
bash scripts/ASTER/train.sh
```
- **Train CRNN**:
```Bash
bash scripts/CRNN/train.sh
```

### 2.4 Check the Training Process
- The training progress are recorded in real-time. Every time you start to train, a recording file will be created in `runs/train`. Inside it are:
  * `eval_tensorboard`: a tensorboard file to record the evaluation process
  * `train_tensorboard`: a tensorboard file to record the training process 
  * `weights`: a file to store the network's weight
  * `log.txt`: all the printed message
  * `log_train.txt`: Created after the first time of evaluation, recording the accuracy and some predictions
  * `train_config.txt`: training configurations
  
#### Check the Tensorboard
- **We also provide you with `Tensorboard` to check the training status**
- To start the tensorboard:
  * If you are coding the `VSCode`, just open the file [lig/utils/logging](lib/utils/logging.py) and the VSCode will give you a message that it can open tensorboard file. How can you resist such a powerful compiler!
  ![img/tensorboard.JPG](../img/tensorboard.JPG)
  * Or you can open the tensorboard with the following commands
  ```Bash
  tensorboard --logdir="runs/train/exp1/train_tensorboard/"
  ```
    * and then open the url `http://localhost:6006` in web broswer. That's not convenient at all.

## 3.Test
- This is used for a single benchmark's evaluation
- Set the test set path `test_data_dir` and the checkpoint path `resume` in any of the `.sh` file. [ASTER for example](scripts/ASTER/test.sh) 
```Bash
bash scripts/ASTER/test.sh
```
- Of course, we provide you wi

## 4.Inference 
- This is used for a single image's inference
- Set the image path `image_path` and the checkpoint path `resume` in any of the `.sh` file. [ASTER for example](scripts/ASTER/inference.sh) 
```Bash
bash scripts/ASTER/inference.sh
```
