'''
A data feeding class. It generates a list of data samples, each of which is
a tuple of a string (image path) and a float (future location), and it defines
a data-fetching method.
'''
import os
from natsort import natsorted
import random
from skimage import io
from torch.utils.data import Dataset, DataLoader
import scipy.io as sciio
import torchvision.transforms as transf
import numpy as np
################## Creating data samples ####################
def create_samples(root,shuffle,nat_sort = False):
    init_names = os.listdir(root) # List all sub-directories in root
    if nat_sort:
        sub_dir_names = natsorted(init_names) # sort directory names in natural order
                                              # (Only for directories with numbers for names)
    else:
        sub_dir_names = init_names

    class_to_ind = {name: name for name in sub_dir_names}
    data_samples = []
    for sub_dir in sub_dir_names: # Loop over all sub-directories
        per_dir = os.listdir(root+'/'+sub_dir) # Get a list of names from sub-dir # i
        if per_dir: # If img_per_dir is NOT empty
            imagelist = []
            wirelist = []
            deplist = []
            dir_count = 0
            for name in per_dir:
                dir_count += 1
                split_name = name.split('_') 
                if split_name[3][-3:]=='jpg':
                  imagelist.append(root + '/' + sub_dir + '/' + str(name))
                elif len(split_name) == 5 and split_name[4]=='sub6.mat':
                  wirelist.append(root + '/' + sub_dir + '/' + str(name))
                elif len(split_name) == 5 and split_name[3] == 'depth':
                  deplist.append(root + '/' + sub_dir + '/' + str(name))
            [l1, l2, l3] = class_to_ind[sub_dir].split('_')
            l1 = float(l1)
            l2 = float(l2)
            l3 = float(l3)
            sample = (imagelist, wirelist, deplist, [l1,l2],l3)
            data_samples.append(sample)
    if shuffle:
        random.shuffle(data_samples)

    return data_samples
#############################################################

class DataFeed(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir, nat_sort = False, transform = None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_samples(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        count = 0
        imagelist = sample[0]
        y=[]
        for imagename in imagelist:
            splitimage = imagename.split('_')
            y.append(float(splitimage[-2]))
        #按大小排列y：
        y.sort()

        wirelesslist = sample[1]
        depthlist = sample[2]

        for i in wirelesslist:
          splitwireless = i.split('_')
          index = y.index(float(splitwireless[-3]))
          if index == 0:
            wireless1 = i
          elif index == 1:
            wireless2 = i


        label = sample[3]
        forsort = sample[4]
        return (wireless1, wireless2, label, forsort)
