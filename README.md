# HandS3C

## 0. environment
The causal-conv1d and mamba-ssm can be downloaded here, pip install /yourpath/xxx install.  \

``` bash
pip install -r requirements.txt
```
All data and model files can be found here 
``` bash
url：https://pan.baidu.com/s/1bl-GJ01Nqs9ZqnNP1VV1KA?pwd=ev6u 
password：ev6u 
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
