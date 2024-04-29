import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import json
import time
import skimage.io as io
from config import cfg
from PIL import Image, ImageDraw

# from utils.fh_utils import *
# from utils.model import HandModel, recover_root, get_focal_pp, split_theta


def mark_keypoints_on_image(original_image, keypoints):
    if keypoints is None:
        original_image = original_image.copy()
        original_image = Image.fromarray(original_image, 'RGB')
        return original_image
    # 确保原始图像和关键点数组都在CPU上
    original_image = original_image.copy()  # 创建一个副本，以免修改原图
    _, img_width, img_height = original_image.shape
    keypoints = keypoints.copy()  # 创建一个副本，以免修改原图
    
    # 将NumPy数组转换为PIL图像
    original_image = Image.fromarray(original_image, 'RGB')
    
    # 创建一个副本，以免修改原图
    marked_image = original_image.copy()
    
    # 创建一个PIL绘图对象
    draw = ImageDraw.Draw(marked_image)
    
    # 关键点的颜色和半径
    keypoint_color = (0, 255, 0)  # 绿色
    keypoint_radius = 5

    # 在图像上绘制关键点
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        draw.ellipse([(x - keypoint_radius, y - keypoint_radius),
                      (x + keypoint_radius, y + keypoint_radius)],
                     fill=keypoint_color, outline=keypoint_color)

    return marked_image

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
    return list(zip(K_list, mano_list, xyz_list))

class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def read_msk(idx, base_path):
    mask_path = os.path.join(base_path, 'training', 'mask',
                             '%08d.jpg' % idx)
    _assert_exist(mask_path)
    return io.imread(mask_path)


class FREI(torch.utils.data.Dataset):
    def __init__(self, transform, data_split="train", target_size=(224, 224)):
        self.transform = transform
        self.data_split = data_split                                    # ["train", "test"]
        self.root_dir = osp.join('..', 'data', 'FREI', 'data')
        if self.data_split == "train":
            self.data_split = 'training'
            self.db_data_anno = load_db_annotation(self.root_dir, 'training')
        else:
            self.data_split = 'evaluation'
            self.db_data_anno = load_db_annotation(self.root_dir, 'evaluation')
        self.target_size = target_size
        if self.data_split != 'train':
            self.eval_result = [[],[]] #[pred_joints_list, pred_verts_list]

    def __len__(self):
        return len(self.db_data_anno)

    def __getitem__(self, idx):
        if self.data_split == "train":
            img = read_img(idx, self.root_dir, self.data_split, sample_version.auto)
        else:
            img = read_img(idx, self.root_dir, self.data_split)
        K, mano, xyz = self.db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)
        img = self.transform(img.astype(np.float32))/255.
        poses = mano[:, :48]
        shapes = mano[:, 48:58]
        poses = np.squeeze(poses, axis=0).astype(np.float32)
        shapes = np.squeeze(shapes, axis=0).astype(np.float32)
        uv_root = mano[:, 58:60].astype(np.float32)
        K = K.astype(np.float32)
        uv = uv.astype(np.float32)
        xyz = xyz.astype(np.float32)
        # normalize to [0,1]
        uv[:,0] /= 224
        uv[:,1] /= 224
        inputs = {'img': img}
        targets = {'joints_img': uv, 'joints3d': xyz, 'cam': K, 'mano_pose': poses, 'mano_shape': shapes}
        meta_info = {'root_joint_cam': uv_root}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        # annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            # annot = annots[cur_sample_idx + n]
            out = outs[n]
            verts_out = out['mesh_coord_cam']
            joints_out = out['joints_coord_cam']
            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())

    def print_eval_result(self, test_epoch):
        output_json_file = osp.join(cfg.result_dir, 'pred{}.json'.format(test_epoch)) 
        
        xyz_pred_list = [x for x in self.eval_result[0]]
        verts_pred_list = [x for x in self.eval_result[1]]

        # save to a json
        with open(output_json_file, 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), output_json_file))