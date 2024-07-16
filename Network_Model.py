import torch
import torch.nn as nn
from build_net import resnet18_mod


class AFE(nn.Module):
    def __init__(self, num_output1):
        super(AFE, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, num_output1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class AFE_Uni_CSI(nn.Module):
        def __init__(self, AFE_feature_size):
            super(AFE_Uni_CSI, self).__init__()
            self.fc1 = nn.Linear(64, 32)
            self.dropout1 = nn.Dropout(0.2)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 16)
            self.dropout2 = nn.Dropout(0.2)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(16, AFE_feature_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.dropout1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.dropout2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return out

class X_MLP_Prediction(nn.Module):
    def __init__(self, input_feature_size, output):
        super(X_MLP_Prediction, self).__init__()
        self.fc1 = nn.Linear(input_feature_size,256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, output)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class Y_PredictNet_MLP(nn.Module):
    def __init__(self, num_output3, final_output):
        super(Y_PredictNet_MLP, self).__init__()
        self.fc1 = nn.Linear(num_output3,32)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(16, final_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class Y_PredictNet_MLP_Uni_CSI(nn.Module):
    def __init__(self, num_output3, final_output):
        super(Y_PredictNet_MLP_Uni_CSI, self).__init__()
        self.fc1 = nn.Linear(num_output3,16)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(16, final_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout2(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


    
class MMFF_net(nn.Module):
    def __init__(self, AFE_feature_size, VFE_feature_size, DFE_feature_size, output_size1, output_size2, h_dim = 16, num_layers = 1):
        super(MMFF_net, self).__init__()
        self.AFE = AFE(AFE_feature_size)
        self.VFE = resnet18_mod(pretrained=True, progress=True, num_classes=VFE_feature_size)
        self.DFE = resnet18_mod(pretrained=True, progress=True, num_classes=DFE_feature_size)
        self._num_layers = num_layers
        self._h_dim = h_dim
        self.AFE_feature_size = AFE_feature_size
        self.VFE_feature_size = VFE_feature_size
        self.DFE_feature_size = DFE_feature_size
        self.multi_modal_feature_size = AFE_feature_size + VFE_feature_size + DFE_feature_size
        self.gru = nn.GRU(self.multi_modal_feature_size, h_dim, num_layers, batch_first=True)
        self.Y_predict_net = Y_PredictNet_MLP(h_dim, output_size1)
        self.X_predict_net = X_MLP_Prediction(AFE_feature_size + VFE_feature_size +DFE_feature_size, output_size2)
        self.squeezefc2 = nn.Sequential(
            nn.Linear(self.multi_modal_feature_size, int(self.multi_modal_feature_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.multi_modal_feature_size/2),int(self.multi_modal_feature_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.multi_modal_feature_size/2), self.multi_modal_feature_size),    
            nn.Sigmoid(),
        )
      

    def forward(self,CSI1,CSI2,RGB1,RGB2,D1,D2):
        # AFE for sub-6 GHz CSI
        wirelessoutput1 = self.AFE(CSI1)
        wirelessoutput2 = self.AFE(CSI2)
        wireless_feature_his = torch.stack([wirelessoutput1,wirelessoutput2],dim=1)  
        # VFE for RGB image
        imageoutput1 = self.VFE(RGB1)
        imageoutput2 = self.VFE(RGB2)
        imageout_feature_his = torch.stack([imageoutput1,imageoutput2],dim=1)     
        # DFE for Depth map
        depoutput1 = self.DFE(D1)
        depoutput2 = self.DFE(D2)
        dep_feature_his = torch.stack([depoutput1,depoutput2],dim=1)
        
        # Entirely preserve all the raw features
        multi_feature = torch.cat((wireless_feature_his, imageout_feature_his, dep_feature_his), 2)
        
        multi_modal_feature_1 = multi_feature[:,0,:]
        multi_modal_feature_2 = multi_feature[:,1,:] 
        
        # Allocate different weights for different sensing modalities
        attention_vector_1 = self.squeezefc2(multi_modal_feature_1)
        attention_vector_2 = self.squeezefc2(multi_modal_feature_2)
        
        # Conduct the modality-wise product
        weight_CSI1=torch.mean(attention_vector_1[:,0:self.AFE_feature_size])
        weight_RGB1=torch.mean(attention_vector_1[:,self.AFE_feature_size:self.AFE_feature_size+self.VFE_feature_size])
        weight_D1=torch.mean(attention_vector_1[:,self.AFE_feature_size+self.VFE_feature_size:self.AFE_feature_size+self.VFE_feature_size+self.DFE_feature_size])
        weight_CSI2=torch.mean(attention_vector_2[:,0:self.AFE_feature_size])
        weight_RGB2=torch.mean(attention_vector_2[:,self.AFE_feature_size:self.AFE_feature_size+self.VFE_feature_size])
        weight_D2=torch.mean(attention_vector_2[:,self.AFE_feature_size+self.VFE_feature_size:self.AFE_feature_size+self.VFE_feature_size+self.DFE_feature_size])
        
        attention_vector_mwise1 = torch.zeros_like(attention_vector_1)
        attention_vector_mwise2 = torch.zeros_like(attention_vector_2)
        attention_vector_mwise1[:, 0:self.AFE_feature_size] = weight_CSI1
        attention_vector_mwise1[:, self.AFE_feature_size:self.AFE_feature_size+self.VFE_feature_size] = weight_RGB1
        attention_vector_mwise1[:, self.AFE_feature_size+self.VFE_feature_size:self.AFE_feature_size+self.VFE_feature_size+self.DFE_feature_size] = weight_D1
        attention_vector_mwise2[:, 0:self.AFE_feature_size] = weight_CSI2
        attention_vector_mwise2[:, self.AFE_feature_size:self.AFE_feature_size+self.VFE_feature_size] = weight_RGB2
        attention_vector_mwise2[:, self.AFE_feature_size+self.VFE_feature_size:self.AFE_feature_size+self.VFE_feature_size+self.DFE_feature_size] = weight_D2
        
        
        multi_modal_feature_afterattention_1 = torch.mul(multi_modal_feature_1, attention_vector_mwise1)
        multi_modal_feature_afterattention_2 = torch.mul(multi_modal_feature_2, attention_vector_mwise2)


        # Concatenate weighted multi-modal feature
        s = torch.stack((multi_modal_feature_afterattention_1, multi_modal_feature_afterattention_2), dim=1)  

            
        # Recurrent Prediction
        state_tuple = None
        output_gru,hidden = self.gru(s, state_tuple)   
        output_mid = output_gru[:,1:]
        y_output = self.Y_predict_net(output_mid)
        y_axis_result = torch.squeeze(y_output,dim=1)
        y_axis_result = torch.squeeze(y_axis_result,dim=1)
        # MLP Prediction
        x_axis_result = self.X_predict_net(multi_modal_feature_afterattention_2)
        x_axis_result =torch.squeeze(x_axis_result,dim=1)
        final_output = [y_axis_result, x_axis_result]
        final_output = torch.stack(final_output, 0)
        final_output = final_output.squeeze()
        return final_output
    

class Uni_Image_net(nn.Module):
        def __init__(self, VFE_feature_size, output_size1, output_size2, embedding_dim=256, h_dim=16, num_layers=1, dropout=0):
            super(Uni_Image_net, self).__init__()
            self.VFE = resnet18_mod(pretrained=True, progress=True, num_classes=VFE_feature_size)
            self.gru = nn.GRU(embedding_dim, h_dim, num_layers, batch_first=True)
            self.Y_predict_net = Y_PredictNet_MLP(h_dim, output_size1)
            self.X_predict_net = X_MLP_Prediction(VFE_feature_size, output_size2)
            self._num_layers = num_layers
            self._h_dim = h_dim

        def forward(self, y1, y2):
            # VFE
            imageoutput1 = self.VFE(y1)
            imageoutput2 = self.VFE(y2)
            # Recurrent Prediction
            imageout_feature_his = torch.stack([imageoutput1,imageoutput2],dim=1)
            state_tuple = None
            output_gru, hidden = self.gru(imageout_feature_his, state_tuple)
            output_mid = output_gru[:, 1:]
            y_output = self.Y_predict_net(output_mid)
            y_axis_result = torch.squeeze(y_output,dim=1)
            y_axis_result = torch.squeeze(y_axis_result,dim=1)
            #  MLP Prediction
            x_axis_result = self.X_predict_net(imageoutput2)
            x_axis_result = torch.squeeze(x_axis_result, 1)
            final_output = [y_axis_result, x_axis_result]
            final_output = torch.stack(final_output, 0)
            final_output = torch.squeeze(final_output)
            return final_output


class Uni_DP_net(nn.Module):
        def __init__(self, DFE_feature_size, output_size1, output_size2, embedding_dim=256, h_dim=16, num_layers=1):
            super(Uni_DP_net, self).__init__()
            self.VFE = resnet18_mod(pretrained=True, progress=True, num_classes=DFE_feature_size)
            self.gru = nn.GRU(embedding_dim, h_dim, num_layers, batch_first=True)
            self.Y_predict_net = Y_PredictNet_MLP(h_dim, output_size1)
            self.X_predict_net = X_MLP_Prediction(DFE_feature_size, output_size2)
            self._num_layers = num_layers
            self._h_dim = h_dim


        def forward(self, z1, z2):
            # Depthmap
            depoutput1 = self.VFE(z1)
            depoutput2 = self.VFE(z2)
            depoutput = torch.stack([depoutput1, depoutput2], dim=1)
            
            # Recurrent Prediction
            state_tuple = None
            output_gru, hidden = self.gru(depoutput, state_tuple)
            output_mid = output_gru[:, 1:]

            y_output = self.Y_predict_net(output_mid)
            y_axis_result = torch.squeeze(y_output,dim=1)
            y_axis_result = torch.squeeze(y_axis_result,dim=1)
           
            # MLP Prediction
            x_axis_result = self.X_predict_net(depoutput2)
            x_axis_result = torch.squeeze(x_axis_result, 1)

            final_output = [y_axis_result, x_axis_result]
            final_output = torch.stack(final_output, 0)
            final_output = torch.squeeze(final_output)
    
            return final_output
        

# Note that the network architecture of Uni-CSI scheme differs from MMFF-Net, Uni-Image scheme, and Uni-Depth-map scheme. The Uni-CSI scheme only predicts the y-coordinate of the vehicle's position. Please refer to the paper for details.
class Uni_CSI_net(nn.Module):
    def __init__(self, AFE_feature_size, output_size1, embedding_dim = 16, h_dim = 4, num_layers = 1):
        super(Uni_CSI_net, self).__init__()
        self.wireless_feature2 = AFE_Uni_CSI(AFE_feature_size)
        self.gru = nn.GRU(embedding_dim, h_dim, num_layers, batch_first=True)
        self.Y_predict_net= Y_PredictNet_MLP(h_dim, output_size1)


    def forward(self, x1,x2):
        # AFE for Sub-6 GHz CSI
        wirelessoutput1 = self.wireless_feature2(x1)
        wirelessoutput2 = self.wireless_feature2(x2)
        wireless_feature_his = torch.stack([wirelessoutput1,wirelessoutput2],dim=1)
        s = wireless_feature_his
        # Recurrent Prediction
        state_tuple = None
        output_gru,hidden = self.gru(s, state_tuple)   
        output_mid = output_gru[:,1:]
        output_mid = output_mid.squeeze()
        y_axis_result = self.Y_predict_net(output_mid)
        return y_axis_result
