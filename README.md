# Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition

                                  Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, and Yu Qiao
                              Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
                                         {kai.wang, xj.peng, db.meng, yu.qiao}@siat.ac.cn

![image](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/sample.png)

## Abstract

Occlusion and pose variations, which can change facial appearance signiﬁcantly, are among two major obstacles for automatic Facial Expression Recognition (FER). Though automatic FER has made substantial progresses in the past few decades, occlusion-robust and pose-invariant issues of FER have received relatively less attention, especially in real-world scenarios.Thispaperaddressesthereal-worldposeandocclusionrobust FER problem with three-fold contributions. First, to stimulate the research of FER under real-world occlusions and variant poses, we build several in-the-wild facial expression datasets with manual annotations for the community. Second, we propose a novel Region Attention Network (RAN), to adaptively capture the importance of facial regions for occlusion and pose variant FER. The RAN aggregates and embeds varied number of region features produced by a backbone convolutional neural network into a compact ﬁxed-length representation. Last, inspired by the fact that facial expressions are mainly deﬁned by facial action units, we propose a region biased loss to encourage high attentionweightsforthemostimportantregions.Weexamineour RAN and region biased loss on both our built test datasets and four popular datasets: FERPlus, AffectNet, RAF-DB, and SFEW. Extensive experiments show that our RAN and region biased loss largely improve the performance of FER with occlusion and variant pose. Our methods also achieve state-of-the-art results on FERPlus, AffectNet, RAF-DB, and SFEW.


![image](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/pipeline_final.png)
## Region Attention Network
we propose the Region Attention Network (RAN), to capture the importance of facial regions for occlusion and pose robust FER. The RAN is comprised of a feature extraction module, a self-attention module, and a relation attention module. The proposed RAN mainly consists of two stages. The first stage is to coarsely calculate the importance of each region by a FC layer conducted on its own feature, which is called self-attention module. The second stage seeks to find more accurate attention weights by modeling the relation between the region features and the aggregated content representation from the first stage, which is called relation-attention module. The latter two modules aim to learn coarse attention weights and refine them with global context, respectively. Given a number of facial regions, our RAN learns attention weights for each region in an end-to-end manner, and aggregates their CNN-based features into a compact fixed-length representation. Besides, the RAN model has two auxiliary effects on the face images. On one hand, cropping regions can enlarge the training data which is important for those insufficient challenging samples. On the other hand, rescaling the regions to the size of original images highlights fine-grain facial features.


<!--
Formally, we denote a face image as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;">, its duplicate as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $${I_1},{I_2}, \cdots ,{I_k}$$" style="border:none;">, and its crops as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;">,  and the backbone CNN as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$r(\cdot;{\theta})$$" style="border:none;">. 
The feature set <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$X$$" style="border:none;"> of <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;"> is defined by: <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$X = [{F_0},{F_1}, \cdots, {F_k}] = [r({I_0};{\theta}),r({I_1};{\theta}), \cdots ,r({I_k};{\theta})]$$" style="border:none;"> where <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\theta$$" style="border:none;"> is the parameter of backbone CNN.
-->

<!--### Relation-attention module-->

### Region Biased Loss
Inspired by the observation that different facial expressions are mainly defined by different facial regions, we make a straightforward constraint on the attention weights of self-attention, *i.e.* region biased loss (RB-Loss). This constraint enforces that one of the attention weights from facial crops should be larger than the original face image with a margin. Formally, the RB-Loss is defined as,<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\mathcal{L}_{RB} = \max\{0, \alpha - (\mu_{max} - \mu_{0})\},$$" style="border:none;"> where <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\alpha$$" style="border:none;"> is a hyper-parameter served as a margin, <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\mu_{0}$$" style="border:none;"> is the attention weight of the copy face image, <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\mu_{max}$$" style="border:none;"> denotes the maximum weight of all facial crops.

## Region Generation 
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/part_generate_v3.png">

## Confused Metrics
<div align="center">
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_affectnet_occlusion-v7.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/affectnet_occlusion-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_affectnet_pose45-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/affectnet_pose45-1.png" width="210" >
 </div>
 
 The confusion matrices of baseline methods and our RAN on the Occlusion- and Pose-FERPlus test sets.

<div align="center">
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_ferplus_occlusion-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/ferplus_occlusion-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_ferplus_pose45-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/ferplus_pose45-1.png" width="210" >
 </div>

The confusion matrices of baseline methods and our RAN on the Occlusion- and Pose-AffectNet test sets.

## What is learned for occlusion and pose variant faces?
<div align="center">
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/w_o_pbl.png" width="420" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/vis_score.png" width="420" >
 </div>
What is learned for occlusion and pose variant faces? In the left, we illustrate the final attention weights with softmax function to better explore our RAN in figure above. To better explore our RAN, we illustrate the final attention weights for several examples with RB-Loss and without RB-Loss, respectively. Occlusion examples are shown in the first two rows, and pose examples in the last two rows.

## Comparison with the state-of-the-art methods
We compare our best results to several stateof-the-art methods on FERPlus, AffectNet, SFEW, and RAFDB.
<div align="center">
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/soat_ferplus.png" width="420" >
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/soat_affectnet.png" width="420" >
</div>

<div align="center">
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/soat_rafdb.png" width="420" >
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/soat_sfew.png" width="420" >
</div>

## Our training codes and collected datasets
You can find occlusion dataset([ferplusocclusion](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/blob/master/FERplus_dir/jianfei_occlusion_list.txt), affectnetocclusion()), pose(>30)([ferpluspose30](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/blob/master/FERplus_dir/pose_30_ferplus_list.txt), [affectnetpose30](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/blob/master/AffectNet_dir/pose_30_affectnet_list.txt)) and pose(>45)([ferpluspose45](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/blob/master/FERplus_dir/pose_30_ferplus_list.txt), affectnetpose45()) list.
