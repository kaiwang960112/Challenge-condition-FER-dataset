import argparse
import os,sys,shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
#from ResNet_MN_Val_all import resnet18, resnet50, resnet101
from val_part_attention import resnet18, resnet34, resnet50, resnet101
from val_part_attention_sample import MsCelebDataset, CaffeCrop
import scipy.io as sio  
import numpy as np
import pdb
import torch._utils
import numpy as np
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir_val', metavar='DIR', default='/media/sdc/kwang/ferplus/different_pose_ferplus/val/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18/checkpoint_40.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')





def get_val_data(list_txt,label_txt,frame_num):

	caffe_crop = CaffeCrop('test')
	txt_path = '/media/sdc/kwang/ferplus/pose_test/test_txt/'
	val_list_file = txt_path+list_txt
	val_label_file = txt_path+label_txt
	#pdb.set_trace()
	val_dataset =  MsCelebDataset(args.img_dir_val, val_list_file, val_label_file, 
	            transforms.Compose([caffe_crop,transforms.ToTensor()]))
	val_loader = torch.utils.data.DataLoader(
	        val_dataset,batch_size=frame_num, shuffle=False,
	num_workers=args.workers, pin_memory=True)

	return val_loader


def main(arch,resume):
    global args
    args = parser.parse_args()
    arch = arch.split('_')[0]
    model = None
    assert(arch in ['resnet18','resnet34','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(end2end=args.end2end)
    if arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if arch == 'resnet50':
        model = resnet50(nverts=nverts,faces=faces,shapeMU=shapeMU,shapePC=shapePC, num_classes=class_num, end2end=args.end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num,\
                extract_feature=True, end2end=end2end)


    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    assert(os.path.isfile(resume))
    #pdb.set_trace()
    checkpoint = torch.load(resume)
    #pdb.set_trace()
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    val_nn_txt = '/media/sdc/kwang/ferplus/pose_test/val_ferplus_mn.txt'
    val_nn_files = open(val_nn_txt,'rb')
    correct = 0
    video_num = 0
    output_task1 = open('ferplus_mn_score.txt','w+')

    for val_nn_file in val_nn_files:
        
        record = val_nn_file.strip().split()
        #pdb.set_trace()
        list_txt = record[0]
        label_txt = record[1]
        frame_num = record[2]
        video_num = video_num +1
        video_name = list_txt
        index_xiahua = video_name.find('_')
        video_name = list(video_name)
        video_name[index_xiahua] = '/'
        #pdb.set_trace()
        video_name = video_name[0:-4]
        video_name = ''.join(video_name)
        print 'video_name',video_name

        val_loader = get_val_data(list_txt,label_txt,int(frame_num))
        
        for i,(input,label) in enumerate(val_loader):
            label = label.numpy()
            input_var = torch.autograd.Variable(input, volatile=True)
            #pdb.set_trace()
            #output, f_need_fix, feature_standard = model(input_var)
            output = model(input_var)
            output_write = output
            output_write =output_write[0]
            output_write = output_write.cpu().data.numpy()
            print 'output_write',output_write
            #pdb.set_trace()
            output_of_softmax = F.softmax(output,dim=1)
            output_of_softmax_ = output_of_softmax.cpu().data.numpy()
            pred_class = np.argmax(output_of_softmax_)
            #output_of_softmax_ = output_of_softmax_[0]
            #output_task1.write(video_name+' '+str(output_of_softmax_[0])+' '+str(output_of_softmax_[1])+' '+str(output_of_softmax_[2])+' '+str(output_of_softmax_[3])+' '+str(output_of_softmax_[4])+' '+str(output_of_softmax_[5])+' '+str(output_of_softmax_[6])+'\n')
            output_task1.write(video_name+' '+str(pred_class)+'\n')
            pred_final = output_of_softmax[0].data.max(0,keepdim=True)[1]
            #pdb.set_trace()
            #pred_final = pred_final.cpu().data.numpy()
            pred_final = pred_final.cpu().numpy()
            if int(label[0]) == int(pred_final[0]):
               correct = correct +1
               print 'predict right label',label[0]
    print 'accuracy', float(correct)/video_num
    print 'correct',correct
    print 'video_num',video_num

if __name__ == '__main__':
    
    #infos = [ ('resnet18_naive', './model/checkpoint_6_654.pth.tar'), 
               #]
	
    infos = [ ('resnet18_naive', '/media/sdc/kwang/ferplus/pose_test/model_best.pth.tar'), 
               ]


    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        main(arch, model_path)
        
        print()