# Proactive Beamforming in mmWave V2I via Multi-Modal Feature Fusion Network (MMFF-Net)

 This simulation code package is mainly used to reproduce the results of the following paper [1]:

 [1] H. Zhang, S. Gao, X. Cheng, and L. Yang, "Integrated Sensing and Communications towards Proactive Beamforming in mmWave V2I via Multi-Modal Feature Fusion (MMFF)," IEEE Trans. Wireless Commun., early access, doi: 10.1109/TWC.2024.3401686.

 If you use this simulation code package in any way, please cite the original paper [1] above.

 Copyright reserved by the Pervasive Connectivity and Networked Intelligence Laboratory (led by Dr. Xiang Cheng), School of Electronics, Peking University, Beijing 100871, China. 


# Dependencies:
1) Python 3.9.7 

2) Pytorch 1.10.2

3) NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

4) Dataset, which can be downloaded at https://pan.baidu.com/s/1zxmSiG3U_zlMzGwymMq1MQ with extraction code my3t.


# Running the code:

1) Ensure there is a folder named Training_dataset for network training; and a folder named Testing_dataset_Normal_Weather, which is used for network testing at normal weather conditions; and a folder named Testing_dataset_Adverse_Weather, which is used for network testing at adverse weather conditions. (For more information on the data structure, see the next section).

2) Set the paths of the training and testing datasets in the script "main.py" (i.e. modify train_dir, test_dir, and test_dir2 to point to your datasets).


3) Run main.py

4) The predicted results of the vehicle's position (given in the form of (x, y)) and the ground truth values will be saved in a .mat file.

# Data Structure:
The script assumes a training and testing datasets structured as a directory of subdirectories, as follows:
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
The name of each sub directory is the coordinate of the vehicle's position at the next time instance, which is the label for training the MMFF-Net. The contents of the subdirectories are the RGB images (e.g., cam_2_27.0268_10.png), depth maps (e.g., cam_2_27.0268_depth_10.mat), and sub-6 GHz channel state information (cam_2_27.0268_10_sub6.mat) collected by RSU at a certain time instant and the time instant. The contents of the subdirectories are inputs for MMFF-Net.

It is noted that there may be some differences in the results of different training processes. 


