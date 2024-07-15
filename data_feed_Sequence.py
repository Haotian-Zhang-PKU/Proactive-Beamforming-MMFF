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
                if split_name[3][-3:]=='png':
                  imagelist.append(root + '/' + sub_dir + '/' + str(name))
                elif len(split_name) == 5 and split_name[4]=='sub6.mat':
                  wirelist.append(root + '/' + sub_dir + '/' + str(name))
                elif len(split_name) == 5 and split_name[3] == 'depth':
                  deplist.append(root + '/' + sub_dir + '/' + str(name))

            [l1, l2] = class_to_ind[sub_dir].split('_')
            l1 = float(l1)
            l2 = float(l2)
            sample = (imagelist, wirelist, deplist, [l1,l2])
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
        y.sort()
        for i in imagelist:
            splitimage = i.split('_')
            index = y.index(float(splitimage[-2]))
            if index == 0:
                img1 = io.imread(i)
            elif index == 1:
                img2 = io.imread(i)

        wirelesslist = sample[1]
        depthlist = sample[2]
    

        for i in wirelesslist:
          splitwireless = i.split('_')
          index = y.index(float(splitwireless[-3]))
          if index == 0:
            wireless1 = i
          elif index == 1:
            wireless2 = i
        
        for i in depthlist:
            splitdep = i.split('_')
            index = y.index(float(splitdep[-3]))
            if index == 0:
                d1 = i
                d1 = sciio.loadmat(d1)['depth']
                # Add noise
                noise1 = np.random.randn(d1.shape[0],d1.shape[1], d1.shape[2]) 
                noise1 = noise1*0.1
                d1 = noise1 + d1
                d1 = transf.ToTensor()(d1)
            elif index == 1:
                d2 = i
                d2 = sciio.loadmat(d2)['depth']
                # Add noise
                noise2 = np.random.randn(d2.shape[0],d2.shape[1], d2.shape[2]) 
                noise2 = noise2*0.1
                d2 = noise2 + d2
                d2 = transf.ToTensor()(d2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = sample[3]
        return (img1, img2, wireless1, wireless2, d1, d2, label)
