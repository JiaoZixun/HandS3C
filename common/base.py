import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_model
import numpy as np

# dynamic dataset import
exec('from ' + cfg.trainset + ' import ' + cfg.trainset)
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, path):
        try:
            model_dict      = model.state_dict()
            weights_dict = torch.load(path, map_location = 'cpu')
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in weights_dict["network"].items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            self.logger.info("\nSuccessful Load Key: {}……\nSuccessful Load Key Num: {}".format(str(load_key)[:500], len(load_key)))
            self.logger.info("\nFail To Load Key: {} ……\nFail To Load Key num: {}".format(str(no_load_key)[:500], len(no_load_key)))
            self.logger.info("\n\033[1;33;44mTips: It is normal that the head part does not load, and it is wrong that the Backbone part does not load.\033[0m")
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            model.load_state_dict(state2)
        start_epoch = weights_dict['epoch'] + 1
        self.logger.info('Load checkpoint from {}'.format(path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        train_dataset = eval(cfg.trainset)(transforms.ToTensor(), "train")
            
        self.itr_per_epoch = math.ceil(len(train_dataset) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=train_dataset, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train')

        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
            start_epoch, model, optimizer = self.load_model(model, optimizer, path)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        self.test_dataset = eval(cfg.testset)(transforms.ToTensor(), "test")
        self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
    
    def load_model(self, model, path):
        try:
            model_dict      = model.state_dict()
            weights_dict = torch.load(path, map_location = 'cpu')
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in weights_dict["network"].items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            self.logger.info("\nSuccessful Load Key: {}……\nSuccessful Load Key Num: {}".format(str(load_key)[:500], len(load_key)))
            self.logger.info("\nFail To Load Key: {} ……\nFail To Load Key num: {}".format(str(no_load_key)[:500], len(no_load_key)))
            self.logger.info("\n\033[1;33;44mTips: It is normal that the head part does not load, and it is wrong that the Backbone part does not load.\033[0m")
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            model.load_state_dict(state2)
        self.logger.info('Load checkpoint from {}'.format(path))
        return model

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        model = self.load_model(model, model_path)
        # ckpt = torch.load(model_path)
        # model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _show(self, outs, cur_sample_idx):
        self.test_dataset.show(outs, cur_sample_idx)

    def _print_eval_result(self, test_epoch):
        self.test_dataset.print_eval_result(test_epoch)