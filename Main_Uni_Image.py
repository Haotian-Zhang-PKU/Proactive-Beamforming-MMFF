import torch
import torch.nn as nn
import torchvision.transforms as transf
from torch.utils.data import DataLoader
from data_feed_Sequence_Uni_Image import DataFeed as DF_I
import numpy as np
import os
import scipy.io
from Network_Model import Uni_Image_net
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

EPOCH = 60
BATCH_SIZE1 = 1
BATCH_SIZE = 1
LR = 0.0001  

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
train_loader = DataLoader(DF_I(train_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        drop_last=True)

net = Uni_Image_net(256, 1, 1)
net.cuda()
loss_func = nn.SmoothL1Loss()
opt = torch.optim.Adam(net.parameters(), lr=LR)
LR_sch = torch.optim.lr_scheduler.MultiStepLR(opt, [15, 30], gamma=0.3, last_epoch=-1)

count = 0
acc_loss = 0
loss_sum = []
step_sum = []
count_epoch = 0
h_state = None
data = []

for epoch in range(EPOCH):
    for tr_count, (img1, img2, label) in enumerate(train_loader):
        count_epoch += 1
        # RGB images
        y1 = img1.cuda()
        y2 = img2.cuda()
        Final_out = net.forward(y1,y2)
        opt.zero_grad()
        # The label is the coordinate of the vehicle's position with a large value, it is not conducive to network learning. Therefore, it is first normalized. During inference stage, the output of the network is subjected to inverse transformation to obtain the coordinates of the vehicle's position.
        label[0] = label[0] * 10
        label[0] = (label[0] - mean_label_X_Train) / std_label_X_Train  
        label[1] = (label[1] - mean_label_Y_Train) / std_label_Y_Train
        label = torch.stack(label)
        label = torch.squeeze(label)
        label = label.cuda()
        
        L = loss_func(Final_out.float(), label.float())
        Final_out = Final_out.float()
        L.backward()
        opt.step()
        batch_loss = L.item()
        acc_loss += batch_loss
        count += 1
        # Output loss every 20 steps
        if count % 20 == 0:
            print('Epoch', epoch, ':', 'batch:', tr_count, 'loss:', batch_loss)
            print('label', label)
            print('Final_out', Final_out)
        loss_sum.append(batch_loss)
        step_sum.append(count_epoch)
    LR_sch.step()


test_dir = '/Testing_dataset_normal_weather'
test_loader = DataLoader(DF_I(test_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=BATCH_SIZE1,
                        shuffle=False,
                        drop_last=True)
acc_loss2 = 0
batch_loss2 = 0
resultx = []
resulty = []
labelx = []
labely = []
resultxreal = []
resultyreal = []
labelreal = []
labelxreal = []
labelyreal = []
difference = []
differencex = []
differencey = []
differencereal = []
differencexreal = []
differenceyreal = []
num = 0
x_l = []
for tr_count, (img1, img2, label) in enumerate(test_loader):
    count_epoch += 1
    y1 = img1.cuda()
    y2 = img2.cuda()
    Final_out = net.forward(y1,y2)
    labelreal = torch.empty(BATCH_SIZE1, 2)
    labelreal[:, 0] = label[0]
    labelreal[:, 1] = label[1]
    label[0] = label[0] * (10)
    label[0] = (label[0] - mean_label_X_Test) / std_label_X_Test
    label[1] = (label[1] - mean_label_Y_Test) / std_label_Y_Test
    # The output of the network is subjected to inverse transformation to obtain the trajectory coordinates of the vehicle.
    Final_outreal = torch.empty(BATCH_SIZE1, 2)
    Final_out = Final_out.view(BATCH_SIZE1, 2)
    Final_outreal[:, 0] = Final_out[:, 0] * std_label_X_Test + mean_label_X_Test
    Final_outreal[:, 0] = Final_outreal[:, 0] / (10)
    Final_outreal[:, 1] = Final_out[:, 1] * std_label_Y_Test + mean_label_Y_Test

    label = torch.stack(label)
    label = label.view(BATCH_SIZE1, 2)
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
        difference.append((Final_out[i][0] - label[i][0]) ** 2 + (Final_out[i][1] - label[i][1]) ** 2)
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
        differencereal.append(
            (Final_outreal[i][0] - labelreal[i][0]) ** 2 + (Final_outreal[i][1] - labelreal[i][1]) ** 2)
        differencexreal.append((Final_outreal[i][0] - labelreal[i][0]) ** 2)
        differenceyreal.append((Final_outreal[i][1] - labelreal[i][1]) ** 2)
        
# Save the prediction results to your folder.
filename = 'MMFF_Uni_Image_result.mat'
scipy.io.savemat("Result/"+filename, mdict={'result_x': resultxreal,'result_y': resultyreal,'label_x': labelxreal, 'label_y': labelyreal})
    
print('---------------------------------------')
print(acc_loss2 / len(test_loader))

