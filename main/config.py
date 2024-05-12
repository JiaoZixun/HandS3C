import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # HO3D, DEX_YCB, FREI(224, 224)
    trainset = 'HO3D'
    testset = 'HO3D'
    
    ## input, output
    input_img_shape = (256, 256)
    
    ## training config
    if trainset == 'HO3D':
        lr_dec_epoch = [10*i for i in range(1,8)]
        end_epoch = 80
        lr = 1e-4
        lr_dec_factor = 0.7
    elif trainset == 'DEX_YCB':
        lr_dec_epoch = [i for i in range(1,20)]
        end_epoch = 20
        lr = 1e-4
        lr_dec_factor = 0.9
    elif trainset == 'Frei':
        lr_dec_epoch = [10*i for i in range(1,10)]
        end_epoch = 100
        lr = 1e-4
        lr_dec_factor = 0.7
    train_batch_size = 64 # per GPU
    lambda_mano_verts = 1e4            
    lambda_mano_joints = 1e4            
    lambda_mano_pose = 10               
    lambda_mano_shape = 0.1             
    lambda_joints_img = 100            
    ckpt_freq = 10

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
    
    def set_dir(self, model_dir, vis_dir, log_dir, result_dir):
        self.model_dir = model_dir
        self.vis_dir = vis_dir
        self.log_dir = log_dir
        self.result_dir = result_dir

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
add_pypath(osp.join(cfg.data_dir, cfg.testset))