import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester
import os
import os.path as osp

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--train_name', type=str, dest='train_name')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    train_file = os.path.join(cfg.output_dir, args.train_name)
    if not os.path.exists(train_file):
        os.mkdir(train_file)
    model_dir = osp.join(train_file, 'model_dump')
    vis_dir = osp.join(train_file, 'vis')
    log_dir = osp.join(train_file, 'log')
    result_dir = osp.join(train_file, 'result')
    make_folder(model_dir)
    make_folder(vis_dir)
    make_folder(log_dir)
    make_folder(result_dir)
    cfg.set_dir(model_dir, vis_dir, log_dir, result_dir)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')
       
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        # # show
        # tester._show(out, cur_sample_idx)
        # evaluate
        tester._evaluate(out, cur_sample_idx)
        cur_sample_idx += len(out)
    
    tester._print_eval_result(args.test_epoch)

if __name__ == "__main__":
    main()