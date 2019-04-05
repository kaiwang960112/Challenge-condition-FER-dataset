This dir contains model and key codes about FERplus dataset.

First you can download ferplus by the Internet, then you can use dlib to crop and align the faces. You can generate your own file list of ferplus dataset.

attention_rank_loss.py is a python code of our method and region attention network.

part_attention_sample.py is our data sample and data load code.

train_attention_rank_loss.py is our training code, some parameters default setting are in this file.

test_rank_loss_attention.py and val_part_attention_sample.py are test codes, note that the network's code don't need to change.

jianfei_occlusion_list, pose_30_ferplus_list and pose_45_ferplus_list are our collected image lists.

Our training model will be share in a link as soon as possible.
