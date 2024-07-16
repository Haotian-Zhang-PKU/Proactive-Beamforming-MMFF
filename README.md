# Proactive Beamforming in mmWave V2I via Multi-Modal Feature Fusion Network (MMFF-Net)

 This simulation code package is mainly used to reproduce the results of the following paper [1]:

 __[1] H. Zhang, S. Gao, X. Cheng, and L. Yang, "Integrated Sensing and Communications towards Proactive Beamforming in mmWave V2I via Multi-Modal Feature Fusion (MMFF)," IEEE Trans. Wireless Commun., early access, doi: 10.1109/TWC.2024.3401686.__

 __If you use this simulation code package in any way, please cite the original paper [1] above.__

 Copyright reserved by the Pervasive Connectivity and Networked Intelligence Laboratory (led by Dr. Xiang Cheng), School of Electronics, Peking University, Beijing 100871, China. 


# Dependencies:
1) Python 3.9.7 

2) Pytorch 1.10.2

3) NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

4) Dataset, which can be downloaded at https://pan.baidu.com/s/1zxmSiG3U_zlMzGwymMq1MQ with extraction code my3t.


# Running the code:

1) Ensure that you have the dataset required for all the schemes. (For more information on the data structure, see the next section).

2) Set the paths of the training and testing datasets in the script "main.py" (i.e. modify train_dir, test_dir, and test_dir2 to point to your datasets).

3) Run Main.py to get the predicted results of MMFF-Net. Run Main_Uni_CSI.py to get the predicted results of the Uni-CSI scheme. Run Main_Uni_Image.py to get the predicted results of the Uni-Image scheme. Run Main_Uni_DP.py to get the predicted results of the Uni-Depth Map scheme. 

4) The predicted results of the vehicle's position (given in the form of (x, y)) and the ground truth values will be saved in a .mat file.

# Data Structure:
The script assumes training and testing datasets structured as a directory of subdirectories, as follows:
```
training_data
  |
  |- x1_y1
  |--cam_2_27.0268_10.png
  |--cam_2_27.0268_10_sub6.mat
  |--cam_2_27.0268_depth_10.mat
  |--cam_2_27.1166_10.png
  |--cam_2_27.1166_10_sub6.mat
  |--cam_2_27.1166_depth_10.mat
  |- x2_y2
  |- x3_y3
  .
  .
  .
  |- xN_yN
 ```
The above dataset structure is applicable to three schemes: MMFF-Net, Uni-Image, and Uni-Depth Map. The name of each sub directory is the coordinate of the vehicle's position at the next time instance, which is the label for training the MMFF-Net. The contents of the subdirectories are the RGB images (e.g., cam_2_27.0268_10.png), depth maps (e.g., cam_2_27.0268_depth_10.mat), and sub-6 GHz CSI (cam_2_27.0268_10_sub6.mat) collected by RSU at a certain time instant and the time instant. The contents of the subdirectories are inputs for MMFF-Net.

```
training_data
  |
  |- Δy1_Δx1_y1
  |--cam_2_27.0268_10.png
  |--cam_2_27.0268_10_sub6.mat
  |--cam_2_27.0268_depth_10.mat
  |--cam_2_27.1166_10.png
  |--cam_2_27.1166_10_sub6.mat
  |--cam_2_27.1166_depth_10.mat
  |- Δy2_Δx2_y2
  |- Δy3_Δx3_y3
  .
  .
  .
  |- ΔyN_ΔxN_yN
 ```
The above dataset structure is applicable to Uni-CSI scheme.  Due to the difficulty in learning the mapping relationship between CSI and vehicle position coordinates by the network, in order to simplify the task difficulty, the Uni CSI scheme only predicts the displacement of the vehicle on the y-axis. Therefore, Δx is 0. The third value is the absolute value of the y-axis coordinate of the vehicle position, which facilitates sorting of the predicted displacement values output by the network. The first two values of each sub directory is the displacement of the x and y coordinates of the vehicle relative to the starting point, which is the label for training the Uni-CSI scheme. The third value is the absolute value of the y-axis coordinate of the vehicle position, which facilitates sorting of the predicted displacement values output by the network. Note that due to the difficulty in learning the mapping relationship between sub-6 GHz CSI and vehicle position coordinates by the network, the Uni-CSI scheme only predicts the displacement of the vehicle on the y-axis to simplify the task difficulty. The contents of the subdirectories are the RGB images (e.g., cam_2_27.0268_10.png), depth maps (e.g., cam_2_27.0268_depth_10.mat), and sub-6 GHz channel state information (cam_2_27.0268_10_sub6.mat) collected by RSU at a certain time instant and the time instant.

It is noted that there may be some differences in the results of different training processes. 

__Acknowledgement__: The code for this study was modified based on the source code provided by Muhammad Alrabeiah, Andrew Hredzak, and Ahmed Alkhateeb for their research: Millimeter Wave Base Stations with Cameras: Vision-Aided Beam and Blockage Prediction. We sincerely appreciate their efforts and help.
