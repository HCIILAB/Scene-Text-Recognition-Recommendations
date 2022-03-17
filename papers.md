# <center>Scene Text Recognition Papers
<h1 align="center">
    <br>
    <img src="img/paper.JPG" >
</h1>

## Contents
- [Methods](#methods)
  - [CTC](#CTC)
  - [Attention](#attention)
  - [Rectification](#rectification)
  - [Language Model](#language)
  - [Dataset](#dataset)
  - [Data Augmentation](#augmentation)
  - [Survey](#survey)
  - [Others](#others)
- [Conference](#conference)
  - [CVPR](#cvpr)
  - [ICCV](#iccv)
  - [ECCV](#eccv)
  - [AAAI](#aaai)
  - [NIPS](#nips)
  - [others](#others1)
- [Journal](#journal)
  - [TPAMI](#tpami)
  - [PR](#pr)

### Methods

<details open>
<summary id='CTC'><strong>CTC</strong></summary>

- **Pattern Recognition-2020,引用数:3**:[Reinterpreting CTC training as iterative fitting](https://www.sciencedirect.com/science/article/pii/S0031320320301953)
  - 探讨CTC数学原理，将CTC Loss解释为交叉熵损失，较为理论
- **ECCV-2020,引用数:2**:[Variational Connectionist Temporal Classification](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_28)
  - 提出变分CTC来增强网络对于非blank符号的学习
- **AAAI-2020,引用数:27**:[GTC: Guided Training of CTC towards Efficient and Accurate Scene Text Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6735)
  - 训练的时候用Attention分支辅助CTC，测试的时候只用CTC

- **IEEE Access-2019,引用数:30**:[Natural Scene Text Recognition Based on Encoder-Decoder Framework](https://ieeexplore.ieee.org/abstract/document/8713973)
  - Attention的解码时候的对齐是没有限制的，故引入CTC对Attention的对齐进行监督
- **ECCV-2018, 引用数:69**:[Synthetically supervised feature learning for scene text recognition](https://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.html)
- **NIPS-2018,引用数:25**:[Connectionist Temporal Classification with Maximum Entropy Regularization](https://papers.nips.cc/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html)
  - 解决CTC中的Spiky Distribution Problem, 使用最大熵来限制CTC学习，较为理论
- **NIPS-2017, 引用数:105**:[Gated recurrent convolution neural network for OCR](https://islab.ulsan.ac.kr/files/announcement/653/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
  - GRCNN
- **Pattern Recognition-2017, 引用数:99**:[Accurate recognition of words in scenes without character segmentation using recurrent neural network](https://www.sciencedirect.com/science/article/pii/S0031320316303314)
- **BMVC-2016,引用数:136**:[STAR-Net: A spatial attention residue network for scene text recognition](http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~scenetext/files/papers/liu_bmvc16.pdf)
  
- **TPAMI-2016,引用数:1497**:[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://ieeexplore.ieee.org/abstract/document/7801919)
  - 场景文字识别开山之作，引入CTC将识别变为一个序列到序列的问题。



</details>

****

<details open>
<summary id='attention'><strong>Attention</strong></summary>

- **AAAI-2022**:[Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://arxiv.org/pdf/2112.08171.pdf)
  * 场景文本超分
- **ICDAR2021**:[Representation and Correlation Enhanced Encoder-Decoder Framework for Scene Text Recognition](https://link.springer.com/chapter/10.1007/978-3-030-86337-1_11)
- **Electronics 2021**: [TRIG: Transformer-Based Text Recognizer with Initial Embedding Guidance](https://www.mdpi.com/2079-9292/10/22/2780)
- **Patter Recognition-2021,引用数:23**:[Master: Multi-aspect non-local network for scene text recognition](https://arxiv.org/pdf/1910.02562.pdf?ref=https://githubhelp.com)
- **arXiv-2021/12/16**:[TRIG: Transformer-Based Text Recognizer with Initial Embedding Guidance](https://arxiv.org/abs/2111.08314)
  - TPS + Transformer Encoder + Attention Decoder的组合。
  - 场景文字超分，引入笔画级别的监督
- **ECCV-2020, 引用数:27**:[Robustscanner: Dynamically enhancing positional clues for robust text recognition](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_9)
- **CVPR-2020, 引用数:42**:[SCATTER: selective context attentional scene text recognizer](https://openaccess.thecvf.com/content_CVPR_2020/html/Litman_SCATTER_Selective_Context_Attentional_Scene_Text_Recognizer_CVPR_2020_paper.html)
  - 多阶段
- **CVPRWorkshop-2020, 引用数:28**:[On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Lee_On_Recognizing_Texts_of_Arbitrary_Shapes_With_2D_Self-Attention_CVPRW_2020_paper.html)
  - Transformer encoder + decoder, 引入自适应的2D位置编码，对于旋转文字，多行文字有较好的鲁棒性
- **AAAI-2020, 引用数：28**:[Textscanner: Reading characters in order for robust scene text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6891)
  - 实例分割
- **AAAI-2020, 引用数：67**:[Decoupled attention network for text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6903)
  - 考虑到注意力机制对于长文本的飘移，采用UNet架构来直接生成注意力图，将注意力图和解码之间解耦开
- **Neural Computing-2020, 引用数:17**:[Adaptive embedding gate for attention-based scene text recognition](https://www.sciencedirect.com/science/article/pii/S0925231219316510)
- **ICCV-2019, 引用数:204**:[What is wrong with scene text recognition model comparisons? dataset and model analysis](https://openaccess.thecvf.com/content_ICCV_2019/html/Baek_What_Is_Wrong_With_Scene_Text_Recognition_Model_Comparisons_Dataset_ICCV_2019_paper.html)
  - 框架型文章，值得一读
- **AAAI-2019，引用数:128**:[Show, attend and read: A simple and strong baseline for irregular text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/4881)
  - 2D Attention
  - 在Attention计算过程中引入门控机制
- **ICDAR-2019, 引用数:45**[NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://ieeexplore.ieee.org/abstract/document/8978180)
  - 使用Transformer encoder和decoder
- **ACM MM-2018, 引用数:44**:[Attention and language ensemble for scene text recognition with convolutional sequence modeling](https://dl.acm.org/doi/abs/10.1145/3240508.3240571)
- **CVPR-2018, 引用数:93**:[Edit probability for scene text recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Bai_Edit_Probability_for_CVPR_2018_paper.html)
  - 现有的attention方法采用最大似然损失函数，本文探讨输出概率分布和预测间的关系
- **AAAI-2018, 引用数:103**:[Char-Net: A character-aware neural network for distorted scene text recognition](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16327)
  - 加入单字检测分支。 
- **AAAI-2018, 引用数:50**:[SEE: Towards Semi-Supervised End-to-End Scene Text Recognition](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16270)
  - 半监督端到端识别。
- **TPAMI-2018, 引用数:329**:[ASTER: An Attentional Scene Text Recognizer
with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027)
  - Attention+Rectification经典之作
- **Neural Computing-2018, 引用数:41**:[Reading scene text with fully convolutional sequence modeling](https://www.sciencedirect.com/science/article/pii/S0925231219301870)
  - Attention算法使用RNN建模，计算复杂并且较难训练，本文使用全卷积网络来捕获全局信息，比BiLSTM更加有效
- **CVPR-2018,引用数:196**:[AON: Towards arbitrarily-oriented text recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Cheng_AON_Towards_Arbitrarily-Oriented_CVPR_2018_paper.html)
  - 关注于不规则文字的识别
- **ICCV-2017, 引用数:290**:[Focusing attention: Towards accurate text recognition in natural images](https://openaccess.thecvf.com/content_iccv_2017/html/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.html)
  - attention存在注意力飘移问题，引入Focusing Network把飘移的注意力抓回来
- **IJCAI-2017, 引用数:124**:[Learning to read irregular text with attention mechanisms](http://personal.psu.edu/duh188/papers/Learning_to_Read_Irregular_Text_with_Attention_Mechanisms.pdf)
- **CVPR-2016, 引用数:370**:[Recursive recurrent nets with attention modeling for OCR in the wild](https://openaccess.thecvf.com/content_cvpr_2016/html/Lee_Recursive_Recurrent_Nets_CVPR_2016_paper.html)


</details>

****
<details open>
<summary id='rectification'><strong>Rectification Model </strong></summary>

- **BMCV-2021**:[An Adaptive Rectification Model for
Arbitrary-Shaped Scene Text Recognition](https://www.bmvc2021-virtualconference.com/assets/papers/1371.pdf)
  - 提出新的矫正方法，在弯曲文本上效果好于TPS和MORAN
- **ICCV-2019, 引用数:77**:[Symmetry-constrained Rectification Network for Scene Text Recognition](https://openaccess.thecvf.com/content_ICCV_2019/htmlYang_Symmetry-Constrained_Rectification_Network_for_Scene_Text_Recognition_ICCV_2019_paper.html)
  - 带限制的矫正网络
- **CVPR-2019, 引用数:165**:[ESIR: End-to-end scene text recognition via iterative image rectification](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhan_ESIR_End-To-End_Scene_Text_Recognition_via_Iterative_Image_Rectification_CVPR_2019_paper.html)
  - 迭代矫正
- **TPAMI-2018, 引用数:329**:[ASTER: An Attentional Scene Text Recognizer
with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027)
  - 引入TPS变换进行矫正
- **Pattern Recognition-2018, 引用数:161**[MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition](https://www.sciencedirect.com/science/article/pii/S0031320319300263)
  - 任意方向矫正，效果比ASTER出色

- **CVPR-2016, 引用数:415**:[Robust Scene Text Recognition With Automatic Rectification](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Robust_Scene_Text_CVPR_2016_paper.html)
  - TPS矫正+Attention


</details>

****

<details open>
<summary id='language'><strong>Language Model</strong></summary>

- **AAAI-2022**:[Visual Semantics Allow for Textual Reasoning Better in Scene Text Recognition](https://arxiv.org/pdf/2112.12916.pdf)
  - 提升语言模型对于任意现状文本的识别能力
- **arXiv-2021/12/1**:[Visual-Semantic Transformer for Scene Text Recognition](https://arxiv.org/pdf/2112.00948.pdf)
- **arXiv-2021/11/30**: [Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features](https://arxiv.org/pdf/2111.15263.pdf)
  - 探讨语言模型和视觉模型如何更好的结合，比肩ABINet，取得SOTA
- **ICCV-2021，引用数:1** [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html)
  - ***VisionLAN***
  - 提出了一个新的遮挡文字数据集
  - 弱监督的将语言模型融入进视觉模型中
- **ICCV-2021，引用数:1** [Joint Visual Semantic Reasoning: Multi-Stage Decoder for Text Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Joint_Visual_Semantic_Reasoning_Multi-Stage_Decoder_for_Text_Recognition_ICCV_2021_paper.html)
  - 多阶段+transformer识别器
  - 引入gumbel softmax，解决视觉到语义不可导问题
- **CVPR-2021，Oral，引用数:1** [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html)
  - ***ABINet***
  - 屠榜作品，在SRN基础进行改良，从人类阅读的角度进行思考
- **CVPR-2020, 引用数:58** [Towards accurate scene text recognition with semantic reasoning networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html)
  - **SRN**
  - 引入Transformer进行语言建模，视觉+语言模型，取得SOTA效果
- **CVPR-2020，引用数:59** [Seed: Semantics enhanced encoder-decoder framework for scene text recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html)
  - **SEED**
  - 在BiLSTM第一个单元前输入语义信息
  - 首次尝试把语言模型引入场景文字识别中

</details>

****


<details open>
<summary id='dataset'><strong>Dataset</strong></summary>

- **CVPR-2020, 引用数:21**:[UnrealText: Synthesizing realistic scene text images from the unreal world](https://arxiv.org/abs/2003.10608)
  - 虚幻引擎来渲染文字图像

- **CVPR-2016, 引用数:979**:[Synthetic data for text localisation in natural images](https://openaccess.thecvf.com/content_cvpr_2016/html/Gupta_Synthetic_Data_for_CVPR_2016_paper.html)
  - SynthText数据集
</details>

- **NIPS-2014, 引用数:737**:[Synthetic data and artificial neural net?works for natural scene text recognition](https://arxiv.org/abs/1406.2227)
  - MJ数据集
****
<details open>
<summary id='augmentation'><strong>Data Augmentation</strong></summary>

- **Arxiv-2021/11/17** [TextAdaIN: Paying Attention to Shortcut Learning in Text Recognizers](https://arxiv.org/abs/2105.03906)
  - 使用Permuted AdaIN进行数据增广
- **CVPR-2020，引用数:32** [Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_Learn_to_Augment_Joint_Data_Augmentation_and_Network_Optimization_for_CVPR_2020_paper.html)
  - 利用仿射变化对图像进行数据增广，有效提升准确率

</details>

****

<details open>
<summary id='survey'><strong>Survey</strong></summary>

- **IJCV-2021, 引用数:156**:[Scene text detection and recognition: The deep learning era](https://link.springer.com/article/10.1007/s11263-020-01369-0)
- **ACM Computing Surveys-2020, 引用数:28**: [Text Recognition in the Wild: A Survey](https://dl.acm.org/doi/abs/10.1145/3440756)
- **TPAMI-2015, 引用数:682**: [Text detection and recognition in imagery: A survey](https://ieeexplore.ieee.org/abstract/document/6945320)
- **Frontiers of Computer Science-2016, 引用数:299**: [Scene text detection and recognition: Recent advances and future trends](https://link.springer.com/article/10.1007%2Fs11704-015-4488-0)

</details>


****

<details open>
<summary id='others'><strong>Others</strong></summary>


- **arXiv-2022**:[Training Protocol Matters: Towards Accurate Scene Text Recognition via Training Protocol Searching](https://arxiv.org/pdf/2203.06696.pdf)
  * 通过搜索训练参数来提升现有模型性能
- **arXiv-2022**:[Text-DIAE: Degradation Invariant Autoencoders for Text Recognition and Document Enhancement](https://arxiv.org/pdf/2203.04814)
- **arXiv-2022**:[Invariant Autoencoders for Text Recognition and Document Enhancement](https://arxiv.org/pdf/2203.03382)]
- **arXiv-2022**:[Towards Open-Set Text Recognition via Label-to-Prototype Learning](https://arxiv.org/pdf/2203.05179)
  * 当测试阶段遇到训练集中没有出现过的字符时，应该如何应对，场景文字识别中的开集问题
- **AAAI-2022**:[Context-based Contrastive Learning for Scene Text Recognition](http://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf)
  - 对比学习用于场景文字识别
- **AAAI-2022**:[FedOCR: Efficient and Secure Federated Learning for Scene Text Recognition](https://federated-learning.org/fl-aaai-2022/Papers/FL-AAAI-22_paper_6.pdf)
  - 联邦学习用于场景文字识别
- **AAAI-2021,引用数:**:[SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117)
  - 图片送入网络前先在颜色上进行矫正
- **arXiv-2021**:[Revisiting Classification Perspective on Scene Text Recognition](https://arxiv.org/abs/2102.10884)
  * 把文本识别当作一个图像分类任务
- **arXiv-2020**:[Hamming OCR: A Locality Sensitive Hashing Neural Network for Scene Text Recognition](https://arxiv.org/abs/2009.10874)
  * 当识别种类数增大时，softmax embedding层就会更大，计算量也就会增大。本文提出使用汉明编码来进行解码，而不是使用one-hot进行解码
- **ICCV Workshop-2021**:[Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Qiu_Meta_Self-Learning_for_Multi-Source_Domain_Adaptation_A_Benchmark_ICCVW_2021_paper.html)
  - 构建了一个多域的中文数据集，定义为Domain Adaptation问题
- **ICCV-2021, 引用数:3**:[Towards the Unseen: Iterative Text Recognition by Distilling from Errors](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Towards_the_Unseen_Iterative_Text_Recognition_by_Distilling_From_Errors_ICCV_2021_paper.html)
  - 重复学习预测错误的样本
- **CVPR-2021, 引用数:1**:[Primitive Representation Learning for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Primitive_Representation_Learning_for_Scene_Text_Recognition_CVPR_2021_paper.html)
  - 表征学习
- **CVPR-2021, 引用数:1**:[What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels]([What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.html))
  - 如果只用真实数据训练识别网络会怎样？
- **CVPR-2021, 引用数:5**:[Sequence-to-Sequence Contrastive Learning for Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.html)
  - 首次在STR中引入对比学习的方法
- **ACM MM-2020, 引用数:6**:[Exploring Font-independent Features for Scene Text Recognition](https://dl.acm.org/doi/abs/10.1145/3394171.3413592)
  - 考虑STR中字体风格的问题，用GAN将字体归一化进行识别
- **CVPR-2020, 引用数:15**:[What Machines See Is Not What They Get: Fooling Scene Text Recognition Models with Adversarial Text Images](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_What_Machines_See_Is_Not_What_They_Get_Fooling_Scene_CVPR_2020_paper.html)
  - 探究STR中的对抗攻击问题
- **CVPR-2020, 引用数:11**:[On Vocabulary Reliance in Scene Text Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Wan_On_Vocabulary_Reliance_in_Scene_Text_Recognition_CVPR_2020_paper.html)
  - 在合成数据集上训练的识别器有字典依赖问题，本文探讨相关解决对策
- **IJCV-2020, 引用数：6**:[Separating content from style using adversarial learning for recognizing text in the wild](https://link.springer.com/article/10.1007/s11263-020-01411-1)
  - 使用对抗生成网络把文字从背景中分离出来进行识别
- **CVPR-2019，引用数:56**:[Sequence-to-sequence domain adaptation network for robust text image recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.html)
  - 探讨STR中的域自适应问题
- **CVPR-2019, 引用数:49**:[Aggregation Cross-Entropy for Sequence Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Aggregation_Cross-Entropy_for_Sequence_Recognition_CVPR_2019_paper.html)
  - 提出了一种全新的聚合交叉熵损失，通过计数的方法做序列识别，速度很快
</details>

### Conference

<details open>
<summary id='cvpr'><strong>CVPR</strong></summary>

- **CVPR-2021, 引用数:5**:[Sequence-to-Sequence Contrastive Learning for Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Aberdam_Sequence-to-Sequence_Contrastive_Learning_for_Text_Recognition_CVPR_2021_paper.html)
  - 首次在STR中引入对比学习的方法
- **CVPR-2021, 引用数:1**:[Primitive Representation Learning for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Primitive_Representation_Learning_for_Scene_Text_Recognition_CVPR_2021_paper.html)
  - 表征学习
- **CVPR-2021, 引用数:1**:[What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels](What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels)
  - 如果只用真实数据训练识别网络会怎样？
- **CVPR-2021，Oral，引用数:1** [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html)
  - ***ABINet***
  - 屠榜作品，在SRN基础进行改良，从人类阅读的角度进行思考
- **CVPR-2020, 引用数:42**:[SCATTER: selective context attentional scene text recognizer](https://openaccess.thecvf.com/content_CVPR_2020/html/Litman_SCATTER_Selective_Context_Attentional_Scene_Text_Recognizer_CVPR_2020_paper.html)
  - 多阶段
- **CVPR-2020, 引用数:58** [Towards accurate scene text recognition with semantic reasoning networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Towards_Accurate_Scene_Text_Recognition_With_Semantic_Reasoning_Networks_CVPR_2020_paper.html)
  - **SRN**
  - 引入Transformer进行语言建模，视觉+语言模型，取得SOTA效果
- **CVPR-2020，引用数:59** [Seed: Semantics enhanced encoder-decoder framework for scene text recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_SEED_Semantics_Enhanced_Encoder-Decoder_Framework_for_Scene_Text_Recognition_CVPR_2020_paper.html)
  - **SEED**
  - 在BiLSTM第一个单元前输入语义信息
  - 首次尝试把语言模型引入场景文字识别中
- **CVPR-2020, 引用数:21**:[UnrealText: Synthesizing realistic scene text images from the unreal world](https://arxiv.org/abs/2003.10608)
  - 虚幻引擎来渲染文字图像
 - **CVPR-2020，引用数:32** [Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_Learn_to_Augment_Joint_Data_Augmentation_and_Network_Optimization_for_CVPR_2020_paper.html)
  - 利用仿射变化对图像进行数据增广，有效提升准确率
- **CVPR-2020, 引用数:15**:[What Machines See Is Not What They Get: Fooling Scene Text Recognition Models with Adversarial Text Images](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_What_Machines_See_Is_Not_What_They_Get_Fooling_Scene_CVPR_2020_paper.html)
  - 探究STR中的对抗攻击问题
- **CVPR-2020, 引用数:11**:[On Vocabulary Reliance in Scene Text Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Wan_On_Vocabulary_Reliance_in_Scene_Text_Recognition_CVPR_2020_paper.html)
- **CVPRWorkshop-2020, 引用数:28**:[On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Lee_On_Recognizing_Texts_of_Arbitrary_Shapes_With_2D_Self-Attention_CVPRW_2020_paper.html)
  - Transformer encoder + decoder, 引入自适应的2D位置编码，对于旋转文字，多行文字有较好的鲁棒性
  - 在合成数据集上训练的识别器有字典依赖问题，本文探讨相关解决对策
- **CVPR-2019, 引用数:165**:[ESIR: End-to-end scene text recognition via iterative image rectification](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhan_ESIR_End-To-End_Scene_Text_Recognition_via_Iterative_Image_Rectification_CVPR_2019_paper.html)
  - 迭代矫正
- **CVPR-2019，引用数:56**:[Sequence-to-sequence domain adaptation network for robust text image recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.html)
  - 探讨STR中的域自适应问题
- **CVPR-2019, 引用数:49**:[Aggregation Cross-Entropy for Sequence Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Aggregation_Cross-Entropy_for_Sequence_Recognition_CVPR_2019_paper.html)
  - 提出了一种全新的聚合交叉熵损失，通过计数的方法做序列识别，速度很快
- **CVPR-2018, 引用数:93**:[Edit probability for scene text recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Bai_Edit_Probability_for_CVPR_2018_paper.html)
  - 现有的attention方法采用最大似然损失函数，本文探讨输出概率分布和预测间的关系
- **CVPR-2018,引用数:196**:[AON: Towards arbitrarily-oriented text recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Cheng_AON_Towards_Arbitrarily-Oriented_CVPR_2018_paper.html)
  - 关注于不规则文字的识别
- **CVPR-2016, 引用数:370**:[Recursive recurrent nets with attention modeling for OCR in the wild](https://openaccess.thecvf.com/content_cvpr_2016/html/Lee_Recursive_Recurrent_Nets_CVPR_2016_paper.html)
- **CVPR-2016, 引用数:415**:[Robust Scene Text Recognition With Automatic Rectification](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Robust_Scene_Text_CVPR_2016_paper.html)
  - TPS矫正+Attention
- **CVPR-2016, 引用数:979**:[Synthetic data for text localisation in natural images](https://openaccess.thecvf.com/content_cvpr_2016/html/Gupta_Synthetic_Data_for_CVPR_2016_paper.html)
  - SynthText数据集

</details>

****

<details open>
<summary id='iccv'><strong>ICCV</strong></summary>

- **ICCV Workshop-2021**:[Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/html/Qiu_Meta_Self-Learning_for_Multi-Source_Domain_Adaptation_A_Benchmark_ICCVW_2021_paper.html)
  - 构建了一个多域的中文数据集，定义为Domain Adaptation问题
- **ICCV-2021, 引用数:3**:[Towards the Unseen: Iterative Text Recognition by Distilling from Errors](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Towards_the_Unseen_Iterative_Text_Recognition_by_Distilling_From_Errors_ICCV_2021_paper.html)
  - 重复学习预测错误的样本
- **ICCV-2021，引用数:1** [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_From_Two_to_One_A_New_Scene_Text_Recognizer_With_ICCV_2021_paper.html)
  - ***VisionLAN***
  - 提出了一个新的遮挡文字数据集
  - 弱监督的将语言模型融入进视觉模型中
- **ICCV-2021，引用数:1** [Joint Visual Semantic Reasoning: Multi-Stage Decoder for Text Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Joint_Visual_Semantic_Reasoning_Multi-Stage_Decoder_for_Text_Recognition_ICCV_2021_paper.html)
  - 多阶段+transformer识别器
  - 引入gumbel softmax，解决视觉到语义不可导问题
- **ICCV-2019, 引用数:204**:[What is wrong with scene text recognition model comparisons? dataset and model analysis](https://openaccess.thecvf.com/content_ICCV_2019/html/Baek_What_Is_Wrong_With_Scene_Text_Recognition_Model_Comparisons_Dataset_ICCV_2019_paper.html)
  - 框架型文章，值得一读
- **ICCV-2019, 引用数:77**:[Symmetry-constrained Rectification Network for Scene Text Recognition](https://openaccess.thecvf.com/content_ICCV_2019/htmlYang_Symmetry-Constrained_Rectification_Network_for_Scene_Text_Recognition_ICCV_2019_paper.html)
  - 带限制的矫正网络
- **ICCV-2017, 引用数:290**:[Focusing attention: Towards accurate text recognition in natural images](https://openaccess.thecvf.com/content_iccv_2017/html/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.html)
  - attention存在注意力飘移问题，引入Focusing Network把飘移的注意力抓回来

</details>

****

<details open>
<summary id='eccv'><strong>ECCV</strong></summary>

- **ECCV-2020, 引用数:27**:[Robustscanner: Dynamically enhancing positional clues for robust text recognition](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_9)
- **ECCV-2020,引用数:2**:[Variational Connectionist Temporal Classification](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_28)
  - 提出变分CTC来增强网络对于非blank符号的学习
- **ECCV-2018, 引用数:69**:[Synthetically supervised feature learning for scene text recognition](https://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Liu_Synthetically_Supervised_Feature_ECCV_2018_paper.html)

</details>

<details open>
<summary id='aaai'><strong>AAAI</strong></summary>

- **AAAI-2022**:[Context-based Contrastive Learning for Scene Text Recognition](http://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf)
  - 对比学习用于场景文字识别
- **AAAI-2022**:[FedOCR: Efficient and Secure Federated Learning for Scene Text Recognition](https://federated-learning.org/fl-aaai-2022/Papers/FL-AAAI-22_paper_6.pdf)
- **AAAI-2022**:[Visual Semantics Allow for Textual Reasoning Better in Scene Text Recognition](https://arxiv.org/pdf/2112.12916.pdf)
  - 提升语言模型对于任意现状文本的识别能力
- **AAAI-2022**:[Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://arxiv.org/pdf/2112.08171.pdf)
  - 场景文字超分，引入笔画级别的监督
- **AAAI-2021,引用数:**:[SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117)
  - 图片送入网络前先在颜色上进行矫正
- **AAAI-2020,引用数:27**:[GTC: Guided Training of CTC towards Efficient and Accurate Scene Text Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6735)
  - 训练的时候用Attention分支辅助CTC，测试的时候只用CTC
- **AAAI-2020, 引用数：28**:[Textscanner: Reading characters in order for robust scene text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6891)
  - 实例分割
- **AAAI-2020, 引用数：67**:[Decoupled attention network for text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6903)
  - 考虑到注意力机制对于长文本的飘移，采用UNet架构来直接生成注意力图，将注意力图和解码之间解耦开
- **AAAI-2019，引用数:128**:[Show, attend and read: A simple and strong baseline for irregular text recognition](https://ojs.aaai.org/index.php/AAAI/article/view/4881)
  - 2D Attention
  - 在Attention计算过程中引入门控机制
- **AAAI-2018, 引用数:103**:[Char-Net: A character-aware neural network for distorted scene text recognition](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16327)
  - 加入单字检测分支。 
- **AAAI-2018, 引用数:50**:[SEE: Towards Semi-Supervised End-to-End Scene Text Recognition](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16270)
  - 半监督端到端识别。
****

</details>

<details open>
<summary id='nips'><strong>NIPS</strong></summary>

- **NIPS-2018,引用数:25**:[Connectionist Temporal Classification with Maximum Entropy Regularization](https://papers.nips.cc/paper/2018/hash/e44fea3bec53bcea3b7513ccef5857ac-Abstract.html)
- **NIPS-2017, 引用数:105**:[Gated recurrent convolution neural network for OCR](https://islab.ulsan.ac.kr/files/announcement/653/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
  - GRCNN
- **NIPS-2014, 引用数:737**:[Synthetic data and artificial neural net?works for natural scene text recognition](https://arxiv.org/abs/1406.2227)
  - MJ数据集
  - 
</details>

****


<details open>
<summary id='others1'><strong>Others</strong></summary>

  - Attention的解码时候的对齐是没有限制的，故引入CTC对Attention的对齐进行监督
- **ACM MM-2018, 引用数:44**:[Attention and language ensemble for scene text recognition with convolutional sequence modeling](https://dl.acm.org/doi/abs/10.1145/3240508.3240571)
- **IJCAI-2017, 引用数:124**:[Learning to read irregular text with attention mechanisms](http://personal.psu.edu/duh188/papers/Learning_to_Read_Irregular_Text_with_Attention_Mechanisms.pdf)
- **BMVC-2016,引用数:136**:[STAR-Net: A spatial attention residue network for scene text recognition](http://cdn.iiit.ac.in/cdn/preon.iiit.ac.in/~scenetext/files/papers/liu_bmvc16.pdf)


</details>

### Journal

<details open>
<summary id='tpami'><strong>TPAMI</strong></summary>

- **TPAMI-2018, 引用数:329**:[ASTER: An Attentional Scene Text Recognizer
with Flexible Rectification](https://ieeexplore.ieee.org/abstract/document/8395027)
  - Attention+Rectification经典之作
- **TPAMI-2016,引用数:1497**:[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://ieeexplore.ieee.org/abstract/document/7801919)
  - 场景文字识别开山之作，引入CTC将识别变为一个序列到序列的问题。


</details>

****


<details open>
<summary id='pr'><strong>Pattern Recognition</strong></summary>

- **Patter Recognition-2021,引用数:23**:[Master: Multi-aspect non-local network for scene text recognition](https://arxiv.org/pdf/1910.02562.pdf?ref=https://githubhelp.com)
- **Pattern Recognition-2020,引用数:3**:[Reinterpreting CTC training as iterative fitting](https://www.sciencedirect.com/science/article/pii/S0031320320301953)
  - 探讨CTC数学原理，将CTC Loss解释为交叉熵损失，较为理论
- **Pattern Recognition-2018, 引用数:161**[MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition](https://www.sciencedirect.com/science/article/pii/S0031320319300263)
  - 任意方向矫正，效果比ASTER出色
- **Pattern Recognition-2017, 引用数:99**:[Accurate recognition of words in scenes without character segmentation using recurrent neural network](https://www.sciencedirect.com/science/article/pii/S0031320316303314)

</details>

### Others

- **IJCV-2021, 引用数:156**:[Scene text detection and recognition: The deep learning era](https://link.springer.com/article/10.1007/s11263-020-01369-0)
- **Neural Computing-2020, 引用数:17**:[Adaptive embedding gate for attention-based scene text recognition](https://www.sciencedirect.com/science/article/pii/S0925231219316510)
- **IEEE Access-2019,引用数:30**:[Natural Scene Text Recognition Based on Encoder-Decoder Framework](https://ieeexplore.ieee.org/abstract/document/8713973)
- **Neural Computing-2018, 引用数:41**:[Reading scene text with fully convolutional sequence modeling](https://www.sciencedirect.com/science/article/pii/S0925231219301870)
  - Attention算法使用RNN建模，计算复杂并且较难训练，本文使用全卷积网络来捕获全局信息，比BiLSTM更加有效
