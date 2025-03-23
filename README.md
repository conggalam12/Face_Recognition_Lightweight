# Method
Here I used 3 models: ShuffleFaceNet, MobileFaceNet and Custom Model for the face recognition problem. <br>These are extremely small models with sizes under 5MB, helping to improve speed faster.You can see the paper here<br>
ShuffleFaceNet : [https://ieeexplore.ieee.org/document/9022220](https://ieeexplore.ieee.org/document/9022220)<br>
MobileFaceNets : [https://arxiv.org/abs/1804.07573](https://arxiv.org/abs/1804.07573)<br>
CusTom Model : [https://link.springer.com/article/10.1007/s00530-022-00973-z](https://link.springer.com/article/10.1007/s00530-022-00973-z)<br>

Dataset: You can refer to the dataset source in this repo <br>
[https://github.com/ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe)

# Model

<img src="https://github.com/user-attachments/assets/7f34823b-8070-4eaa-875c-3c353b6b4867" alt="Image Description" width="600" height="auto">


# Survey
| Model | Size | 
|--------|----------|
| Shuffle   | 2.0 MB   |
| Mobile   | 4.0 MB   | 
| CusTom   | 1.5 MB   | 

# Loss Function
I use 4 loss functions: ArcMarginProduct, AddMarginProduct, CosFace and FocalLoss. <br>You can customize the loss in the config file, change the GPU as well as change the dataset and threshold<br>
ArcFace : [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)<br>
CosFace : [https://arxiv.org/abs/1801.09414](https://arxiv.org/abs/1801.094140)

# Develop
I also applied facial recognition to the ncnn framework using C++ to optimize speed <br />
You can see how to build and use with [NCNN](https://github.com/Tencent/ncnn) <br />
Build [opencv](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) C++ <br />

