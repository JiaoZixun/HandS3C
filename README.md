# HandS3C: 3D Hand Mesh Reconstruction with State Space Spatial Channel Attention from RGB images

Our paper has been accepted as oral by ICASSP2025 and can be found here![here](https://ieeexplore.ieee.org/document/10887953)

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
