# HandSSCA

## 0. environment
The causal-conv1d and mamba-ssm can be downloaded here, pip install /yourpath/xxx install.  \

``` bash
pip install -r requirements.txt
```
## 1. test
The data and models can be downloaded here. \

Unzip the data to /yourpath/HandSSCA/data/xxx/data and change the path in xxx.py for the corresponding data processing. (xxx: HO3D, DEX_YCB, FREI)  \

The download model is placed in the /yourpath/HandSSCA/output/xxx/model_dump    \

### HO3D
``` bash
cd main
python test.py --gpu 0 --test_epoch 80 --train_name HO3D_HandSSCA   # (--test_epoch Indicates the number of epochs, --train_name Indicates the name of the folder in output)
```
### DEX_YCB
``` bash
python test.py --gpu 0 --test_epoch 20 --train_name DEX_HandSSCA
```
### FREI
It should be noted that the FREI dataset image size of (224, 224) requires changes to the code for testing.
``` bash
python test.py --gpu 0 --test_epoch 100 --train_name FREI_HandSSCA
```
