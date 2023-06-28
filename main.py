# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import toml
import argparse
from   pprint import pprint

import torch
from   torch.utils.data import DataLoader

import utils
from   utils import CONFIG
from   trainer import Trainer
from   dataloader.image_file import ImageFileTrain
from   dataloader.data_generator import DataGenerator
from   dataloader.prefetcher import Prefetcher
import wandb
import warnings
warnings.filterwarnings("ignore")

def main():

    # Train or Test
    if CONFIG.phase.lower() == "train":
        # set distributed training
        if CONFIG.dist:
            CONFIG.gpu = CONFIG.local_rank
            torch.cuda.set_device(CONFIG.gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            CONFIG.world_size = torch.distributed.get_world_size()

        # Create directories if not exist.
        if CONFIG.local_rank == 0:
            utils.make_dir(CONFIG.log.logging_path)
            utils.make_dir(CONFIG.log.tensorboard_path)
            utils.make_dir(CONFIG.log.checkpoint_path)
            if CONFIG.wandb:
                wandb.init(project="mam", config=CONFIG, name=CONFIG.version)
    
        # Create a logger
        logger, tb_logger = utils.get_logger(CONFIG.log.logging_path,
                                             CONFIG.log.tensorboard_path,
                                             logging_level=CONFIG.log.logging_level)
    
        train_dataset = DataGenerator(phase='train')

        if CONFIG.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CONFIG.model.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=CONFIG.data.workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      drop_last=True)
        train_dataloader = Prefetcher(train_dataloader)

        trainer = Trainer(train_dataloader=train_dataloader,
                          test_dataloader=None,
                          logger=logger,
                          tb_logger=tb_logger)
        trainer.train()
    else:
        raise NotImplementedError("Unknown Phase: {}".format(CONFIG.phase))

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--config', type=str, default='config/gca-dist.toml')
    parser.add_argument('--local_rank', type=int, default=0)
    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
    CONFIG.local_rank = args.local_rank
    
    # Train
    main()
