import torch
import random
import argparse
import os
import numpy as np
from torch.backends import cudnn
from config.config import cfg, cfg_from_file, cfg_from_list
import sys
import pprint
import json
from utils.utils import print_networks
from utils.neptune import setup_neptune_logging
from utils.set_random_seed import set_random_seed

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--method', dest='method',
                        help='set the method to use',
                        default='McDalNet', type=str)
    parser.add_argument('--task', dest='task',
                        help='closed | partial | open',
                        default='closed', type=str)
    parser.add_argument('--distance_type', dest='distance_type',
                        help='set distance type in McDalNet',
                        default='L1', type=str)

    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name',
                        default='log', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train(args, logger=None):
    set_random_seed(cfg.SEED)
    # Only method 'SymmNetsV1' has been used in this work.
    if args.method == 'SymmNetsV1':
        if cfg.DATASET.DATASET == 'Digits':
            raise NotImplementedError
        else:
            from solver.SymmNetsV1_solver import SymmNetsV1Solver as Solver
            from models.resnet_SymmNet import resnet as Model
            from data.prepare_data import generate_dataloader as Dataloader
        feature_extractor, classifier = Model()

        if torch.cuda.is_available():
            feature_extractor.cuda()
            classifier.cuda()
        net = {'feature_extractor': feature_extractor, 'classifier': classifier}
        print_networks(net['feature_extractor'], False)
        print_networks(net['classifier'], False)
    else:
        raise NotImplementedError("Currently don't support the specified method: %s." % (args.method))

    dataloaders = Dataloader()

    # initialize solver
    train_solver = Solver(net, dataloaders, logger)

    # train
    train_solver.solve()
    print('Finished!')


if __name__ == '__main__':
    cudnn.benchmark = True
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.resume is not None:
        cfg.RESUME = args.resume
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name + cfg.DATASET.DATASET + '_' + cfg.DATASET.SOURCE_NAME + '2' + cfg.DATASET.VAL_NAME + '_' + args.method + '_' +args.distance_type + args.task
    if args.distance_type is not None:
        cfg.MCDALNET.DISTANCE_TYPE = args.distance_type

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)
    if not os.path.isdir(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    log = open(os.path.join(cfg.SAVE_DIR, 'log.txt'), 'a')
    log.write("\n")
    log.write(json.dumps(cfg) + '\n')
    log.close()

    if cfg.NEPTUNE_LOGGING:
        logger = setup_neptune_logging(cfg.NEPTUNE.PROJECT, cfg.NEPTUNE.API_TOKEN)
        logger['cfg'] = cfg
        logger['sys/tags'].add([cfg.EXP_NAME[:4]])
    else:
        logger = None

    train(args, logger)

    if cfg.NEPTUNE_LOGGING:
        logger.stop()