import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine_synapse import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from config_setting_synapse import setting_config

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
from torch.utils.data.distributed import DistributedSampler


def main(config):

    print('#----------Creating logger----------#')
    # print(config.work_dir)
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    # print("log_dir:")
    # print(log_dir)
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # print("checkpoint_dir:")
    # print(checkpoint_dir)
    # print()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    resume_model = os.path.join(current_dir,
        'best-epoch.pth')

    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = 0  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)

    
    model = torch.nn.DataParallel(model.cuda(), device_ids=[gpu_ids], output_device=gpu_ids)


    print('#----------Preparing dataset----------#')
    test_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if config.distributed else None
    test_loader = DataLoader(test_dataset,
                            batch_size=1,  # if config.distributed else config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            sampler=test_sampler,
                            drop_last=True)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    mean_dice, mean_hd95 = val_one_epoch(
        test_dataset,
        test_loader,
        model,
        1,
        logger,
        config,
        test_save_path=outputs,
        val_or_test=True
    )



if __name__ == '__main__':
    config = setting_config
    main(config)