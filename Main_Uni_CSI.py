import torch
import torch.nn as nn
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from data_feed_Sequence_Uni_CSI import DataFeed as DF_CSI
import numpy as np
import h5py as h5
import math
import os
import scipy.io
from Network_Model import Uni_CSI_net
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

EPOCH = 60
BATCH_SIZE1 = 1
BATCH_SIZE = 1
LR = 0.0003

def processdata(data, x):
    _range = np.max(data) - np.min(data)
    return (x - np.min(data)) / _range

img_resize = transf.Resize((224, 224))   #改变输入特征
img_norm = transf.Normalize(mean=(0.306, 0.281, 0.251), std=(0.016, 0.0102, 0.013))    #对像素值进行归一化处理
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
    img_resize,
    transf.ToTensor(),
    img_norm]
)

mean_label_X_Train = torch.tensor(17.8233)
mean_label_Y_Train=torch.tensor(0)
max_label_X_Train=torch.tensor(35.3772) 
min_label_X_Train=torch.tensor(0.2693)  
std_label_X_Train=torch.tensor(10.1736)
std_label_Y_Train=torch.tensor(0)

mean_label_X_Test = torch.tensor(17.8233)
mean_label_Y_Test=torch.tensor(41.7075)
max_label_X_Test=torch.tensor(35.3772)  
min_label_X_Test=torch.tensor(0.2693)   
std_label_X_Test=torch.tensor(10.1736)
std_label_Y_Test=torch.tensor(0)

def normalization(data, x):
    _range = np.max(data) - np.min(data)
    return (x - np.min(data)) / _range


def processdata(input):
    input = torch.sort(input=input, descending=False).values
    mean_a = torch.mean(input)
    std_a = torch.std(input)
    out = (input - mean_a) / std_a
    return out




# Data pre-processing:
img_resize = transf.Resize((224, 224))   
img_norm = transf.Normalize(mean=(0.306, 0.281, 0.251), std=(0.016, 0.0102, 0.013))  
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
    img_resize,
    transf.ToTensor(),
    img_norm]
)


# Read Training Dataset
train_dir = '/Training_dataset'
train_loader = DataLoader(DF_CSI(train_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        drop_last=True)


net = Uni_CSI_net(16,1)
net.cuda()
loss_func = nn.SmoothL1Loss()
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-8)
LR_sch = torch.optim.lr_scheduler.MultiStepLR(opt, [30], gamma=0.3, last_epoch=-1)

count=0
acc_loss=0
loss_sum=[]
step_sum=[]
count_epoch=0
h_state = None
data=[]

for epoch in range(EPOCH):
    for tr_count, (wireless1, wireless2, label, forsort) in enumerate(train_loader):
        count_epoch+=1
        # CSI file1
        filename1 = wireless1[0]
        f = h5.File(filename1, 'r')
        key1 = list(f.keys())
        data = f[key1[0]]
        channels = data[:]  # Wireless data field
        X1 = channels.view(np.double)
        X1 = X1.astype(np.float32)    
        wireless1 = torch.from_numpy(X1)
        wireless1 = wireless1.cuda().to(torch.float32)
        
        # CSI file2
        filename2 = wireless2[0]
        # Read MATLAB structure:
        f = h5.File(filename2, 'r')
        key1 = list(f.keys())
        raw_data = f[key1[0]]
        channels = raw_data[:]  # Wireless data field
        X2 = channels.view(np.double)
        X2 = X2.astype(np.float32)
        wireless2 = torch.from_numpy(X2)
        wireless2 = wireless2.cuda().to(torch.float32)
        
        Final_out = net(wireless1,wireless2)
        opt.zero_grad()
        
        # The label is the coordinate of the vehicle's position with a large value, it is not conducive to network learning. Therefore, it is first normalized. During inference stage, the output of the network is subjected to inverse transformation to obtain the coordinates of the vehicle's position.
        label[0] = (label[0] - min_label_X_Train) / (max_label_X_Train-min_label_X_Train) 
        label = label[0]
        label = label.cuda()
        L = loss_func(Final_out.float(), label.float())
        Final_out =Final_out.float()
        L.backward()
        opt.step()
        batch_loss = L.item()
        acc_loss += batch_loss
        count += 1
        #  Output loss every 20 steps
        if count % 20 == 0:
            print(Final_out)
            print('label', label)
            print('Epoch', epoch, ':','batch:', tr_count, 'loss:', batch_loss)
        loss_sum.append(batch_loss)
        step_sum.append(count_epoch)
    LR_sch.step()


test_dir = '/Testing_dataset'
test_loader = DataLoader(DF_CSI(test_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        drop_last=True)
acc_loss2=0
batch_loss2=0

resulty=[]
labely=[]

resultyreal=[]
labelyreal=[]

differencey=[]
differenceyreal=[]

num=0
x_l=[]
forsortm=np.zeros(len(test_loader))
for tr_count, (wireless1, wireless2, label, forsort) in enumerate(test_loader):

    count_epoch += 1
    # CSI file1
    filename1 = wireless1[0]
    f = h5.File(filename1, 'r')
    key1 = list(f.keys())
    data = f[key1[0]]
    channels = data[:]  # Wireless data field
    X1 = channels.view(np.double)
    X1 = X1.astype(np.float32)    
    wireless1 = torch.from_numpy(X1)
    wireless1 = wireless1.cuda().to(torch.float32)
    
    # CSI file2
    filename2 = wireless2[0]
    # Read MATLAB structure:
    f = h5.File(filename2, 'r')
    key1 = list(f.keys())
    raw_data = f[key1[0]]
    channels = raw_data[:]  # Wireless data field
    X2 = channels.view(np.double)
    X2 = X2.astype(np.float32)
    wireless2 = torch.from_numpy(X2)
    wireless2 = wireless2.cuda().to(torch.float32)
    Final_out = net.forward(wireless1,wireless2)

    labelreal = torch.empty(BATCH_SIZE, 1)
    labelreal[:, 0] = label[0]
    label[0] = (label[0] - min_label_X_Test) / (max_label_X_Test-min_label_X_Test)  
    label = label[0]
    # The output of the network is subjected to inverse transformation to obtain the trajectory coordinates of the vehicle.
    Final_outreal = torch.empty(BATCH_SIZE, 1)
    Final_out = Final_out.view(BATCH_SIZE, 1)
    Final_outreal[:, 0] = Final_out[:, 0] * (max_label_X_Test-min_label_X_Test) + min_label_X_Test
    label = label.view(BATCH_SIZE, 1)
    label = label.cuda()
    print('Output of Net:', Final_out, 'Label of Net:', label)
    print('Actual Predicted Result:',Final_outreal, 'Actual label:',labelreal)
    L = loss_func(Final_out.float(), label.float())
    batch_loss2 = L.item()
    acc_loss2 += batch_loss2

    Final_out = Final_out.detach().cpu().numpy().tolist()
    label = label.detach().cpu().numpy().tolist()
    for i in range(BATCH_SIZE):
        resulty.append(Final_out[i][0])
        labely.append(label[i][0])
        differencey.append((Final_out[i][0] - label[i][0]) ** 2)
    
    Final_outreal = Final_outreal.detach().cpu().numpy().tolist()
    labelreal = labelreal.detach().cpu().numpy().tolist()
    for i in range(BATCH_SIZE):
        resultyreal.append(Final_outreal[i][0])
        labelyreal.append(labelreal[i][0])
        differenceyreal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2)
        forsortm[tr_count]=forsort


# Sort the relative changes in the y-coordinate output by the network based on the values used for sorting in the labels.
sorted_id = sorted(range(len(forsortm)), key=lambda k: forsortm[k], reverse=True)
forsortm.sort()
tempsort1=np.zeros(len(forsortm),dtype=float)
for j in range(len(forsortm)):
    tempsort1[j] = resultyreal[sorted_id[j]]
    
resultyreal=tempsort1
print('---------------------------------------')
print(acc_loss2/len(test_loader))


rmsey = math.sqrt(sum(differencey)/len(test_loader))
print(rmsey)
rmsey = math.sqrt(sum(differenceyreal)/len(test_loader))
print(rmsey)


# Convert relative changes in y-coordinate values to absolute y-coordinate values.
temprho=23.8841
yreal=[]
count=0
for i in range(len(resultyreal)):
    yreal.append(temprho+resultyreal[count])
    count=count+1
# Save the prediction results to your folder.
filenameCSI = 'MMFF_Uni_CSI_result.mat'
scipy.io.savemat("/Result/"+filenameCSI, mdict={'CSI_data': yreal})       
print(yreal)

