import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import cv2
def load_imgs(img_dir, image_list_file, label_file):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    imgs_forth = list()
    imgs_fifth = list()
    imgs_sixth = list()
    imgs_seventh = list()
    imgs_eigth = list()
    max_label = 0
    with open(image_list_file, 'r') as imf:
        with open(label_file, 'r') as laf:
            
            
            for line in imf:
                
                space_index = line.find(' ')
                
                video_name = line[0:space_index]  # name of video
                img_count = line[space_index+1:]  # number of frames in video
                
                video_path = os.path.join(img_dir, video_name)# video_path is the path of each video
                ###  for sampling triple imgs in the single video_path  ####
                
                img_lists = listdir(video_path)
                # pdb.set_trace()
                record = laf.readline().strip().split()
                img_lists.sort()  #  sort files by ascending
                # pdb.set_trace()
                img_path_first =    video_path+'/'+img_lists[0]
                img_path_second = video_path + '/'+img_lists[0]
                img_path_third = video_path + '/'+img_lists[0]
                img_path_forth = video_path + '/'+img_lists[0]
                img_path_fifth = video_path + '/' + img_lists[0]
                img_path_sixth = video_path + '/' + img_lists[0]
                img_path_seventh = video_path + '/' + img_lists[0]
                img_path_eigth = video_path + '/' + img_lists[0]
                #pdb.set_trace()
                label = int(record[0])            
                imgs_first.append((img_path_first,label))
                imgs_second.append((img_path_second,label))
                imgs_third.append((img_path_third,label))
                imgs_forth.append((img_path_forth,label))
                imgs_fifth.append((img_path_fifth,label))
                imgs_sixth.append((img_path_sixth,label))
                #pdb.set_trace()
                imgs_seventh.append((img_path_seventh,label))
                imgs_eigth.append((img_path_eigth,label))

                ###  return multi paths in a single video  #####

           
                #print 'record[0],record[1],record[2]',record[0],record[1],record[2]
                
    return imgs_first,imgs_second,imgs_third, imgs_forth, imgs_fifth, imgs_sixth, imgs_seventh, imgs_eigth

class MsCelebDataset(data.Dataset):
    def __init__(self, img_dir, image_list_file, label_file, transform=None):
        self.imgs_first, self.imgs_second, self.imgs_third, self.imgs_forth, self.imgs_fifth, self.imgs_sixth, self.imgs_seventh, self.imgs_eigth = load_imgs(img_dir, image_list_file, label_file)
        self.transform = transform

    def __getitem__(self, index):
        # pdb.set_trace()
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        # img_first = cv2.imread(path_first)
        # center down
        height = img_first.size[0]
        width = img_first.size[1]
        box = (int(0.125*width), int(0.25*height), int(0.875*width), int(1*height))
        img_first = img_first.crop(box)
        # img_first = img_first[int(0.25*height):int(1*height),int(0.125*width):int(0.875*width)]
        if self.transform is not None:
            img_first = self.transform(img_first)
        # top left
        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        box = (int(0*width), int(0*height), int(0.75*width), int(0.75*height))
        # img_second = img_second[int(0*height):int(0.75*height),int(0*width):int(0.75*width)]
        img_second = img_second.crop(box)
        if self.transform is not None:
            img_second = self.transform(img_second)

        # top right

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        box = (int(0.25*width), int(0*height), int(1*width), int(0.75*height))
        # img_third = img_third[int(0*height):int(0.75*height),int(0.25*width):int(1*width)]
        img_third = img_third.crop(box)
        if self.transform is not None:
            img_third = self.transform(img_third)

        # first center crop

        path_forth, target_forth = self.imgs_forth[index]
        img_forth = Image.open(path_forth).convert("RGB")
        box = (int(0.05*width), int(0.05*height), int(0.95*width), int(0.95*height))
        # img_forth = img_forth[int(0.05*height):int(0.95*height),int(0.05*width):int(0.95*width)]
        img_forth = img_forth.crop(box)
        if self.transform is not None:
            img_forth = self.transform(img_forth)





        # second center crop
        path_fifth, target_fifth = self.imgs_fifth[index]
        img_fifth = Image.open(path_fifth).convert("RGB")
        box = (int(0.1*width), int(0.1*height), int(0.9*width), int(0.9*height))
        # img_fifth = img_fifth[int(0.1*height):int(0.9*height),int(0.1*width):int(0.9*width)]
        img_fifth = img_fifth.crop(box)
        if self.transform is not None:
            img_fifth = self.transform(img_fifth)



       # img original
        path_sixth, target_sixth = self.imgs_sixth[index]
        img_sixth = Image.open(path_sixth).convert("RGB")
        box = (int(0.125*width), int(0.125*height), int(0.875*width), int(0.875*height))
        img_sixth = img_sixth.crop(box)       
        if self.transform is not None:
            img_sixth = self.transform(img_sixth)
        # pdb.set_trace()


        # center top

        path_seventh, target_seventh = self.imgs_seventh[index]
        img_seventh = Image.open(path_seventh).convert("RGB")
        box = (int(0.125*width), int(0*height), int(0.875*width), int(0.75*height))
        img_seventh = img_seventh.crop(box)
        if self.transform is not None:
            img_seventh = self.transform(img_seventh)

        # third center crop

        path_eigth, target_eigth = self.imgs_eigth[index]
        img_eigth = Image.open(path_eigth).convert("RGB")
        #box = (int(0.125*width), int(0.125*height), int(0.875*width), int(0.875*height))
        #img_eigth = img_eigth.crop(box)
        if self.transform is not None:
            img_eigth = self.transform(img_eigth)
        #pdb.set_trace()

        return img_first, target_first ,img_second,target_second,img_third,target_third,img_forth,target_forth, img_fifth, target_fifth, img_sixth, target_sixth, img_seventh, target_seventh, img_eigth, target_eigth
    def __len__(self):
        return len(self.imgs_first)




class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        crop_size = 110
        crop_height = crop_width = crop_size
        crop_center_y_offset = 15
        crop_center_x_offset = 0
        if self.phase == 'train':
            scale_aug = 0.02
            trans_aug = 0.01
        else:
            scale_aug = 0.0
            trans_aug = 0.0
        
        # computed parameters
        randint = rd.randint
        scale_height_diff = (randint(0,1000)/500-1)*scale_aug
        crop_height_aug = crop_height*(1+scale_height_diff)
        scale_width_diff = (randint(0,1000)/500-1)*scale_aug
        crop_width_aug = crop_width*(1+scale_width_diff)


        trans_diff_x = (randint(0,1000)/500-1)*trans_aug
        trans_diff_y = (randint(0,1000)/500-1)*trans_aug


        center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
                 (img.height/2 + crop_center_y_offset)*(1+trans_diff_y))

        
        if center[0] < crop_width_aug/2:
            crop_width_aug = center[0]*2-0.5
        if center[1] < crop_height_aug/2:
            crop_height_aug = center[1]*2-0.5
        if (center[0]+crop_width_aug/2) >= img.width:
            crop_width_aug = (img.width-center[0])*2-0.5
        if (center[1]+crop_height_aug/2) >= img.height:
            crop_height_aug = (img.height-center[1])*2-0.5

        crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
                    center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

        mid_img = img.crop(crop_box)
        res_img = img.resize( (final_width, final_height) )
        return res_img
