import torch
import torch.nn as nn
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from data_feed_Sequence import DataFeed as DataFeedN
from data_feed_Sequence_Noise import DataFeed as DataFeedTest
import numpy as np
import h5py as h5
import scipy.io
import os
from Network_Model import MMFF_net

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

EPOCH = 50            
BATCH_SIZE1 = 1
BATCH_SIZE = 1
sub6_antenna = 8
LR = 0.0003                 


def normalization(data, x):
    _range = np.max(data) - np.min(data)
    return (x - np.min(data)) / _range

def processdata(input):
    input = torch.sort(input=input, descending=False).values
    mean_a = torch.mean(input)
    std_a = torch.std(input)
    out = (input - mean_a) / std_a
    return out


mean_label_X_Train = torch.tensor(101.4286)
mean_label_Y_Train=torch.tensor(41.7075)
max_label_X_Train=torch.tensor(110) 
min_label_X_Train=torch.tensor(90)  
std_label_X_Train=torch.tensor(6.1107)
std_label_Y_Train=torch.tensor(10.1736)

mean_label_X_Test = torch.tensor(98.9031)
mean_label_Y_Test=torch.tensor(41.7075)
max_label_X_Test=torch.tensor(110)  
min_label_X_Test=torch.tensor(90)   
std_label_X_Test=torch.tensor(7.0127)
std_label_Y_Test=torch.tensor(10.1736)



# Data pre-processing:
img_resize = transf.Resize((224, 224))   
img_norm = transf.Normalize(mean=(0.306, 0.281, 0.251), std=(0.016, 0.0102, 0.013))    
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
    img_resize,
    transf.ToTensor(),
    img_norm]
)

#Read Training Dataset
train_dir = '/Training_dataset'
train_loader = DataLoader(DataFeedN(train_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE,
                        shuffle= False,
                        drop_last = True)


net = MMFF_net(16,256,256,1,1)
net.cuda()
loss_func = nn.SmoothL1Loss()
opt = torch.optim.Adam(net.parameters(), lr=LR)
LR_sch = torch.optim.lr_scheduler.MultiStepLR(opt, [15,30], gamma=0.3, last_epoch=-1)   #15 30 0.3


count=0
acc_loss=0
loss_sum=[]
step_sum=[]
count_epoch=0
h_state = None
data=[]
decay = 1e-8


for epoch in range(EPOCH):
    for tr_count, (img1, img2, wireless1, wireless2, d1, d2, label) in enumerate(train_loader):
        count_epoch+=1
        
        # CSI file1
        filename1 = wireless1[0]
        f = h5.File(filename1, 'r')
        key1 = list(f.keys())
        raw_data = f[key1[0]]
        channels = raw_data[:]  # Wireless data field
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
        
        # RGB image
        RGB1 = img1.cuda()
        RGB2 = img2.cuda()
        
        # Preprocess Depth Map
        d1 = (d1-torch.min(d1)) / (torch.max(d1)-torch.min(d1))
        d2 = (d2-torch.min(d2)) / (torch.max(d2)-torch.min(d2))
        d1 = d1.type(torch.cuda.FloatTensor)
        d2 = d2.type(torch.cuda.FloatTensor)   
        D1 = d1.cuda()
        D2 = d2.cuda()
        
        Final_out = net.forward(wireless1,wireless2,RGB1,RGB2,D1,D2)
        opt.zero_grad()
        ## The label is the coordinate of the vehicle's position with a large value, it is not conducive to network learning. Therefore, it is first normalized. During inference stage, the output of the network is subjected to inverse transformation to obtain the coordinates of the vehicle's position.
        label[0] = label[0] * 10
        label[0] = (label[0] - mean_label_X_Train) / std_label_X_Train  
        label[1] = (label[1] - mean_label_Y_Train) / std_label_Y_Train
        label = torch.stack(label)
        label = torch.squeeze(label)
        label = label.cuda()
        L = loss_func(Final_out.float(), label.float())
        L.backward()
        opt.step()
        batch_loss = L.item()
        acc_loss += batch_loss
        count += 1
        # Output loss every 20 steps
        if count % 20 == 0:
            print('label:', label)
            print('result:', Final_out)
            print('Epoch', epoch, ':','batch:', tr_count, 'loss:', batch_loss)
        loss_sum.append(batch_loss)
        step_sum.append(count_epoch)
    LR_sch.step()



print('*******************Testing Dataset: Normal Weather*******************')

# Read testing dataset
test_dir = '/Testing_dataset_normal_weather'
test_loader = DataLoader(DataFeedN(test_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE1,
                        shuffle=False,
                        drop_last = True)
acc_loss=0
batch_loss=0
resultx=[]
resulty=[]
labelx=[]
labely=[]
resultxreal=[]
resultyreal=[]
labelreal=[]
labelxreal=[]
labelyreal=[]
difference=[]
differencex=[]
differencey=[]
differencereal=[]
differencexreal=[]
differenceyreal=[]
num=0
x_l=[]
# net.eval()
with torch.no_grad():
    for tr_count, (img1, img2, wireless1, wireless2,  d1, d2, label) in enumerate(test_loader):
        count_epoch += 1
        #CSI file1
        filename1 = wireless1[0]
        f = h5.File(filename1, 'r')
        key1 = list(f.keys())
        raw_data = f[key1[0]]
        channels = raw_data[:]  # Wireless data field
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
        
        # RGB images
        RGB1 = img1.cuda()
        RGB2 = img2.cuda()
        
        # Preprocess depth map
        d1 = d1.type(torch.cuda.FloatTensor)
        d2 = d2.type(torch.cuda.FloatTensor)
        d1 = (d1-torch.min(d1)) / (torch.max(d1)-torch.min(d1))
        d2 = (d2-torch.min(d2)) / (torch.max(d2)-torch.min(d2))
        d1 = d1.type(torch.cuda.FloatTensor)
        d2 = d2.type(torch.cuda.FloatTensor)
        D1 = d1.cuda()
        D2 = d2.cuda()
        
        Final_out = net.forward(wireless1, wireless2, RGB1, RGB2, D1, D2)

        labelreal = torch.empty(BATCH_SIZE1, 2)
        labelreal[:,0] = label[0]
        labelreal[:,1] = label[1]
        label[0] = label[0] * 10
        label[0] = (label[0] - mean_label_X_Test) / std_label_X_Test
        label[1] = (label[1] - mean_label_Y_Test) / std_label_Y_Test
        
        # The output of the network is subjected to inverse transformation to obtain the trajectory coordinates of the vehicle.
        Final_outreal = torch.empty(BATCH_SIZE1,2)
        Final_out = Final_out.view(BATCH_SIZE1,2)
        
        Final_outreal[:,0] = Final_out[:,0] * std_label_X_Test + mean_label_X_Test
        Final_outreal[:,0] = Final_outreal[:,0] / 10
        Final_outreal[:,1] = Final_out[:,1] * std_label_Y_Test + mean_label_Y_Test
        
        label = torch.stack(label)
        label = label.view(BATCH_SIZE1,2)
        label = label.cuda()
        print('Output of Net:', Final_out, 'Label of Net:', label)
        print('Actual Predicted Result:',Final_outreal, 'Actual label:',labelreal)
        L = loss_func(Final_out.float(), label.float())
        batch_loss = L.item()
        acc_loss += batch_loss
  
        Final_out = Final_out.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        for i in range(BATCH_SIZE1):
            resultx.append(Final_out[i][0])
            resulty.append(Final_out[i][1])
            labelx.append(label[i][0])
            labely.append(label[i][1])
            difference.append((Final_out[i][0]-label[i][0])**2+(Final_out[i][1]-label[i][1])**2)
            differencex.append((Final_out[i][0] - label[i][0]) ** 2)
            differencey.append((Final_out[i][1] - label[i][1]) ** 2)
        x_l.append(num)
        
        Final_outreal = Final_outreal.detach().cpu().numpy().tolist()
        labelreal = labelreal.detach().cpu().numpy().tolist()
        for i in range(BATCH_SIZE1):
            resultxreal.append(Final_outreal[i][0])
            resultyreal.append(Final_outreal[i][1])
            labelxreal.append(labelreal[i][0])
            labelyreal.append(labelreal[i][1])
            print(Final_outreal[i][0])
            print(labelreal[i][0])
            differencereal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2 + (Final_outreal[i][1] - labelreal[i][1]) ** 2)
            differencexreal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2)
            differenceyreal.append((Final_outreal[i][1] - labelreal[i][1]) ** 2)

#save the result
filename_MMFF_vali_normal = 'MMFF_vali_normal.mat'
scipy.io.savemat("Result/"+filename_MMFF_vali_normal, mdict={'result_x': resultxreal,'result_y': resultyreal,'label_x': labelxreal, 'label_y': labelyreal})
    
print('---------------------------------------')
print(acc_loss/len(test_loader))

print('*******************Change Testing Dataset: Adverse Weather*******************')

test_dir2 = '/Testing_dataset_adverse_weather'
test_loader2 = DataLoader(DataFeedTest(test_dir2, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE1,
                        shuffle=False,
                        drop_last = True)
acc_loss2=0
batch_loss2=0
resultx=[]
resulty=[]
labelx=[]
labely=[]
resultxreal=[]
resultyreal=[]
labelreal=[]
labelxreal=[]
labelyreal=[]
difference=[]
differencex=[]
differencey=[]
differencereal=[]
differencexreal=[]
differenceyreal=[]
num=0
x_l=[]

with torch.no_grad():
    for tr_count, (img1, img2, wireless1, wireless2,  d1, d2, label) in enumerate(test_loader2):
        count_epoch += 1
        #CSI file1 (with noise)
        filename1 = wireless1[0]
        f = h5.File(filename1, 'r')
        key1 = list(f.keys())
        raw_data = f[key1[0]]
        channels = raw_data[:]  # Wireless data field
        X1 = channels.view(np.double)
        X1 = X1.astype(np.float32)    
        wireless1 = torch.from_numpy(X1)
        wireless1 = wireless1.cuda().to(torch.float32)
        
        # CSI file2 (with noise)
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
        
        # RGB images (Decreased light intensity)
        RGB1 = img1.cuda()
        RGB2 = img2.cuda()
        
        # Preprocess depth map (with noise)
        d1 = d1.type(torch.cuda.FloatTensor)
        d2 = d2.type(torch.cuda.FloatTensor)
        d1 = (d1-torch.min(d1)) / (torch.max(d1)-torch.min(d1))
        d2 = (d2-torch.min(d2)) / (torch.max(d2)-torch.min(d2))
        d1 = d1.type(torch.cuda.FloatTensor)
        d2 = d2.type(torch.cuda.FloatTensor)
        D1 = d1.cuda()
        D2 = d2.cuda()
        
        Final_out = net.forward(wireless1, wireless2, RGB1, RGB2, D1, D2)

        labelreal = torch.empty(BATCH_SIZE1, 2)
        labelreal[:,0] = label[0]
        labelreal[:,1] = label[1]
        label[0] = label[0] * 10
        label[0] = (label[0] - mean_label_X_Test) / std_label_X_Test
        label[1] = (label[1] - mean_label_Y_Test) / std_label_Y_Test
      
        Final_outreal = torch.empty(BATCH_SIZE1,2)
        Final_out = Final_out.view(BATCH_SIZE1,2)
        
        Final_outreal[:,0] = Final_out[:,0] * std_label_X_Test + mean_label_X_Test
        Final_outreal[:,0] = Final_outreal[:,0] / 10
        Final_outreal[:,1] = Final_out[:,1] * std_label_Y_Test + mean_label_Y_Test
     
        
        label = torch.stack(label)
        label = label.view(BATCH_SIZE1,2)
        label = label.cuda()
        print('Output of Net:', Final_out, 'Label of Net:', label)
        print('Actual Predicted Result:',Final_outreal, 'Actual label:',labelreal)
        L = loss_func(Final_out.float(), label.float())
        
        batch_loss2 = L.item()
        acc_loss2 += batch_loss2
       
        Final_out = Final_out.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        for i in range(BATCH_SIZE1):
            resultx.append(Final_out[i][0])
            resulty.append(Final_out[i][1])
            labelx.append(label[i][0])
            labely.append(label[i][1])
            difference.append((Final_out[i][0]-label[i][0])**2+(Final_out[i][1]-label[i][1])**2)
            differencex.append((Final_out[i][0] - label[i][0]) ** 2)
            differencey.append((Final_out[i][1] - label[i][1]) ** 2)
        x_l.append(num)
        
        Final_outreal = Final_outreal.detach().cpu().numpy().tolist()
        labelreal = labelreal.detach().cpu().numpy().tolist()
        for i in range(BATCH_SIZE1):
            resultxreal.append(Final_outreal[i][0])
            resultyreal.append(Final_outreal[i][1])
            labelxreal.append(labelreal[i][0])
            labelyreal.append(labelreal[i][1])
            print(Final_outreal[i][0])
            print(labelreal[i][0])
            differencereal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2 + (Final_outreal[i][1] - labelreal[i][1]) ** 2)
            differencexreal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2)
            differenceyreal.append((Final_outreal[i][1] - labelreal[i][1]) ** 2)


#save the result
filename_MMFF_vali_w_noise = 'MMFF_vali_w_noise.mat'
scipy.io.savemat("Result/"+filename_MMFF_vali_w_noise, mdict={'result_x': resultxreal,'result_y': resultyreal,'label_x': labelxreal, 'label_y': labelyreal})
    

print('---------------------------------------')
print(acc_loss2/len(test_loader2))