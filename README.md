# Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition

                                  Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, and Yu Qiao<sup>*</sup>
                              Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

![image](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/sample.png)

## Abstract

Occlusion and pose variations, which can change facial appearance signiﬁcantly, are among two major obstacles for automatic Facial Expression Recognition (FER). Though automatic FER has made substantial progresses in the past few decades, occlusion-robust and pose-invariant issues of FER have received relatively less attention, especially in real-world scenarios.Thispaperaddressesthereal-worldposeandocclusionrobust FER problem with three-fold contributions. First, to stimulate the research of FER under real-world occlusions and variant poses, we build several in-the-wild facial expression datasets with manual annotations for the community. Second, we propose a novel Region Attention Network (RAN), to adaptively capture the importance of facial regions for occlusion and pose variant FER. The RAN aggregates and embeds varied number of region features produced by a backbone convolutional neural network into a compact ﬁxed-length representation. Last, inspired by the fact that facial expressions are mainly deﬁned by facial action units, we propose a region biased loss to encourage high attentionweightsforthemostimportantregions.Weexamineour RAN and region biased loss on both our built test datasets and four popular datasets: FERPlus, AffectNet, RAF-DB, and SFEW. Extensive experiments show that our RAN and region biased loss largely improve the performance of FER with occlusion and variant pose. Our methods also achieve state-of-the-art results on FERPlus, AffectNet, RAF-DB, and SFEW.


## Region Attention Network
![image](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/pipeline_final.png)

The proposed RAN mainly consists of two stages. The first stage is to coarsely calculate the importance of each region by a FC layer conducted on its own feature, which is called self-attention module. The second stage seeks to find more accurate attention weights by modeling the relation between the region features and the aggregated content representation from the first stage, which is called relation-attention module.

Formally, we denote a face image as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;">, its duplicate as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $${I_1},{I_2}, \cdots ,{I_k}$$" style="border:none;">, and its crops as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;">,  and the backbone CNN as <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$r(\cdot;{\theta})$$" style="border:none;">. 

The feature set <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$X$$" style="border:none;"> of <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$I$$" style="border:none;"> is defined by: <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$X = [{F_0},{F_1}, \cdots, {F_k}] = [r({I_0};{\theta}),r({I_1};{\theta}), \cdots ,r({I_k};{\theta})]$$" style="border:none;"> where <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$\theta$$" style="border:none;"> is the parameter of backbone CNN.

### Self-attention module

### Relation-attention module

### Region Biased Loss

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
<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_affectnet_occlusion-v7.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/affectnet_occlusion-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/baseline_affectnet_pose45-1.png" width="210" >

<img src="https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/affectnet_pose45-1.png" width="210" >
 </div>


