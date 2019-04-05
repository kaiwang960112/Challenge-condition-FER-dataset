# Region Attention Networks for Pose and Occlusion Robust Facial Expression Recognition
Kai Wang, Xiaojiang Peng, Jianfei Yang, Debin Meng, and Yu Qiao

![image](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/raw/master/figs/w_o_pbl.png)

## Abstract

Abstract—Occlusion and pose variations, which can change facial appearance signiﬁcantly, are among two major obstacles for automatic Facial Expression Recognition (FER). Though automatic FER has made substantial progresses in the past few decades, occlusion-robust and pose-invariant issues of FER have received relatively less attention, especially in real-world scenarios.Thispaperaddressesthereal-worldposeandocclusionrobust FER problem with three-fold contributions. First, to stimulate the research of FER under real-world occlusions and variant poses, we build several in-the-wild facial expression datasets with manual annotations for the community. Second, we propose a novel Region Attention Network (RAN), to adaptively capture the importance of facial regions for occlusion and pose variant FER. The RAN aggregates and embeds varied number of region features produced by a backbone convolutional neural network into a compact ﬁxed-length representation. Last, inspired by the fact that facial expressions are mainly deﬁned by facial action units, we propose a region biased loss to encourage high attentionweightsforthemostimportantregions.Weexamineour RAN and region biased loss on both our built test datasets and four popular datasets: FERPlus, AffectNet, RAF-DB, and SFEW. Extensive experiments show that our RAN and region biased loss largely improve the performance of FER with occlusion and variant pose. Our methods also achieve state-of-the-art results on FERPlus, AffectNet, RAF-DB, and SFEW.


## Region Attention Network
