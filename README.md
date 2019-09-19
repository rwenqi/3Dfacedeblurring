# Face Video Deblurring via 3D Facial Priors

Wenqi Ren, Jiaolong Yang, Senyou Deng, David Wipf, Xiaochun Cao, and Xin Tong

Existing face deblurring methods only consider single frames and do not account for facial structure and identity information. These methods struggle to deblur face videos that exhibit significant pose variations and misalignment. In this paper we propose a novel face video deblurring network capitalizing on 3D facial priors. The model consists of two main branches: i) a face video deblurring subnetwork based on an encoder-decoder architecture, and ii) a 3D face reconstruction and rendering branch for predicting 3D priors of salient facial structures and identity knowledge. These structures encourage the deblurring branch to generate sharp faces with detailed structures. Our method leverages both image intensity and high-level identity information derived from the reconstructed 3D faces to deblur the input face video. Extensive experimental results demonstrate that the proposed algorithm performs favorably against the state-of-the-art methods.

# Installation
Install required packages:

tf_mesh_renderer: Please refer to `./faceReconstruction/tf_mesh_renderer_installation.txt` and [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction)


# Running
1. run `preprocess/generateAlignments.m` to generate `training_set/` or `testing_set/` and the image list of `datalist_train.txt` or `datalist_test.txt`. 

2. run `facePointDetection/demo_landmark.m` to generate `dataset/[videos_folder_list]/face/`，and `dataset/[videos_folder_list]/bbox.txt`, where "bbox.txt" is the detected five key points of faces. For more information about face key points detection, please refer to [Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm). You can also try other state-of-the-art face key points detection method to generate the five key points.

3. run `demo.py` to generate the 3D facial relevant vector under `training_set/[videos_folder_list]/face/`.

4. run `demo_render.py` to generate the rendered face results under `dataset/[videos_folder_list]/face_render/`.  

5. run run_model.py to train a model or test your own videos. For example 
```
# Training
python run_model.py --phase=train --gpu=0 --datalist=<'./datalist_train.txt'>
# Testing
python run_model.py --phase=test --gpu=0 --datalist=<'./datalist_test.txt'> --input_path=<'./testing_set'> --output_path=<'./testing_results>' 
```


# Pre-trained models
Download the [pre-trained 3d rendering model](https://drive.google.com/drive/folders/1Y4h37OigbHvZyNGd4NvbZXR1NUzI9qPS?usp=sharing), and put files under `faceReconstruction/network/` and `faceReconstruction/BFM/`

Download the [pre-trained deblurring model](https://drive.google.com/drive/folders/1xaPaLQnRFnHFVgOrhZ_8RSYymp-Q9FqJ?usp=sharing), and put files under `3Dfacedeblurring/checkpoints`.   

We also train a network using blurry face videos synthesized by 9 consecutive frames. The trained model can be downloaded from [pre-trained deblurring model by 9 frames]().
  
# Model
The parameter of `--face` means the different configuration of network. `--face=both` means using both rendered face and facial vector. `--face=render` denotes only using rendered face. `--face=coeff` represents that only using the face vector without 3D rendered face. 

# Citations
Please cite this paper in your publications if it helps your research:    
@inproceedings{Ren-ICCV-2019,    
&nbsp;author = {Ren, Wenqi and Yang, Jiaolong and Deng, Senyou and Wipf, David and Cao, Xiaochun and Tong, Xin},   
&nbsp;title = {Face Video Deblurring via 3D Facial Priors},    
&nbsp;booktitle = {IEEE International Conference on Computer Vision},   
&nbsp;year = {2019}   
}

# Acknowledgement
Our deblurring branch is based on the implementation of [Tao et al. CVPR 2018](https://github.com/jiangsutx/SRN-Deblur), Thanks for the sharing.

# Reference
Shuochen Su, Mauricio Delbracio, Jue Wang, Guillermo
Sapiro, Wolfgang Heidrich, and Oliver Wang. Deep video
deblurring for hand-held cameras. In CVPR, 2017.

Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep
multi-scale convolutional neural network for dynamic scene
deblurring. In CVPR, 2017.

Grigorios G Chrysos and Stefanos Zafeiriou. Deep face deblurring. In CVPRW, 2017

Xin Tao, Hongyun Gao, Xiaoyong Shen, Jue Wang, and Jiaya Jia. Scale-recurrent network for deep image deblurring.
In CVPR, 2018.

Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde
Jia, and Xin Tong. Accurate 3d face reconstruction with
weakly-supervised learning: From single image to image set.
In CVPR Workshop on Analysis and Modeling of Faces and
Gestures, 2019. 

