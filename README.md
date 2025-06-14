# HandS3C: 3D Hand Mesh Reconstruction with State Space Spatial Channel Attention from RGB images

## Our paper has been accepted as oral by ICASSP2025 and can be found [here](https://ieeexplore.ieee.org/document/10887953)

```
\title{HandS3C: 3D Hand Mesh Reconstruction \\ with State Space Spatial Channel Attention \\ from RGB images
}

% \author{\IEEEauthorblockN{1\textsuperscript{st} Zixun Jiao}
% \IEEEauthorblockA{
% \textit{Xi’an Polytechnic University}\\
% Xi’an, China \\
% 0000-0001-9438-6124}
% \and
% \IEEEauthorblockN{2\textsuperscript{nd} Xihan Wang}
% \IEEEauthorblockA{
% \textit{Xi’an Polytechnic University}\\
% Xi’an, China \\
% xihanwang@xpu.edu.cn}
% \and
% \IEEEauthorblockN{3\textsuperscript{rd} Zhaoqiang Xia}
% \IEEEauthorblockA{
% \textit{Northwestern Polytechnical University}\\
% Xi’an, China \\
% zxia@nwpu.edu.cn}
% \and
% \IEEEauthorblockN{4\textsuperscript{th} Lianhe Shao}
% \IEEEauthorblockA{
% \textit{Xi’an Polytechnic University}\\
% Xi’an, China \\
% shaolianhe@xpu.edu.cn}
% \and
% \IEEEauthorblockN{5\textsuperscript{th} Quanli Gao}
% \IEEEauthorblockA{
% \textit{Xi’an Polytechnic University}\\
% Xi’an, China \\
% gaoquanli@xpu.edu.cn}
% }

\author{
Zixun Jiao$^{1}$ \qquad Xihan Wang$^{1}$ \qquad Zhaoqiang Xia$^{2}$
\qquad Lianhe Shao$^{1}$ \qquad Quanli Gao$^{1,*}$
\\
$^{1}$Xi’an Polytechnic University, Xi’an, China,
$^{2}$Northwestern Polytechnical University, Xi’an, China \\
$^{*}$corresponding author
}
```

## Introduction
Effective receptive fields (from left to right: FPN, Self-Attention, State Space Attention (SSA) and State-Space Spatial-Channel Attention (S3C)).
![Effective receptive fields](https://github.com/JiaoZixun/HandS3C/blob/main/fig1.jpg "Effective receptive fields")



Attention characterization map. It is worth noting that the S3C module is able to obtain sufficient hand information in the hand interaction edge region and the occlusion region. And the method expands the range of the receptive field so that it can capture the global feature mapping. (Orange is the occluded region and red is the hand interaction edge region)
![Effective receptive fields](https://github.com/JiaoZixun/HandS3C/blob/main/fig6.jpg "Effective receptive fields")


## 0. environment
The causal-conv1d and mamba-ssm can be downloaded here, pip install /yourpath/xxx install.  \

``` bash
pip install -r requirements.txt
```
All data and model files can be found here 
``` bash
url：https://pan.baidu.com/s/1gYYI1y8wSCPr5QAhRGnksQ?pwd=ckol 
password：ckol 
```

## 1. test
The data and models can be downloaded here. \

Unzip the data to /yourpath/HandS3C/data/xxx/data and change the path in xxx.py for the corresponding data processing. (xxx: HO3D, DEX_YCB, FREI)  \

The download model is placed in the /yourpath/HandS3C/output/xxx/model_dump    \

### HO3D
``` bash
cd main
python test.py --gpu 0 --test_epoch 80 --train_name HO3D_HandS3C   # (--test_epoch Indicates the number of epochs, --train_name Indicates the name of the folder in output)
```
### DEX_YCB
``` bash
python test.py --gpu 0 --test_epoch 20 --train_name DEX_HandS3C
```
### FREI
It should be noted that the FREI dataset image size of (224, 224) requires changes to the code for testing.
``` bash
python test.py --gpu 0 --test_epoch 100 --train_name FREI_HandS3C
```


## Reference
``` bash
@INPROCEEDINGS{10887953,
  author={Jiao, Zixun and Wang, Xihan and Xia, Zhaoqiang and Shao, Lianhe and Gao, Quanli},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={HandS3C: 3D Hand Mesh Reconstruction with State Space Spatial Channel Attention from RGB images}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Hands;Solid modeling;Three-dimensional displays;Computational modeling;Computer architecture;Signal processing;Feature extraction;Computational efficiency;Speech processing;Image reconstruction;3D Hand Mesh Reconstruction;Deep Learning;Effective Receptive Field;Human-computer Interaction;State Space Model},
  doi={10.1109/ICASSP49660.2025.10887953}}
```
