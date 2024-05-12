import torch
import torch.nn as nn
from torch.nn import functional as F

from config import cfg
from nets.backbone import FPN
from nets.S3C import S3C
from nets.regressor import Regressor



class Model(nn.Module):
    def __init__(self, backbone, s3c, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.S3C = s3c
        self.regressor = regressor
        
    def forward(self, inputs, targets, meta_info, mode):
        feats = self.backbone(inputs['img'])
        feats = self.S3C(feats)
        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)
       
        if mode == 'train':
            # loss functions
            loss = {}
            loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
            loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
            loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])
            return loss
        else:
            # test output
            out = {}
            out['joints_coord_img'] = preds_joints_img[0]
            out['mano_pose'] = pred_mano_results['mano_pose_aa']
            out['mano_shape'] = pred_mano_results['mano_shape']
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            out['manojoints2cam'] = pred_mano_results['manojoints2cam'] 
            out['mano_pose_aa'] = pred_mano_results['mano_pose_aa']
            return out
    
def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    s3c = S3C(hidden_dim=256, W=32, attn_drop_rate=0)         # State Space channel Attention   Frei:28   HO3D: 32  
    regressor = Regressor()
    if mode == 'train':
        regressor.apply(init_weights)
    model = Model(backbone, s3c, regressor)
    return model