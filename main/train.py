import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--train_name', type=str, dest='train_name')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.train_name:
        print("Warning: No name specified for the training. Using default name.")
        args.train_name = 'default_training'
    
    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
   
    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    args.continue_train = True
    cfg.set_args(args.gpu_ids, args.continue_train)
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
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    writer = SummaryWriter(log_dir)
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            global_steps = epoch * len(trainer.batch_generator) + itr
            for k,v in loss.items():
                writer.add_scalar(f'Loss/train_{k}', v.detach(), global_steps)
            
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        if (epoch+1)%cfg.ckpt_freq== 0 or epoch+1 == cfg.end_epoch:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch+1)
        

if __name__ == "__main__":
    main()