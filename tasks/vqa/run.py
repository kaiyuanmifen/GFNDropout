# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from cfgs.base_cfgs import Cfgs
from core.exec import Execution
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--CTYPE', dest='ctype',
                        choices=["Gaussian", "Bernoulli"],
                        default="Gaussian", type=str)

    parser.add_argument('--DP_TYPE', dest='dp_type',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--CONCRETE', dest='concretedp',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--LEARNPRIOR', dest='learn_prior',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--ARM', dest='ARM',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--add_noise', dest='add_noise',
                        choices=[1, 0],
                        default=0, type=int)

    parser.add_argument('--noise_scalar', dest='noise_scalar',
                        default=0.0, type=float)

    parser.add_argument('--UNCERTAINTY_SAMPLE', dest='uncertainty_sample',
                        default=20, type=int)

    parser.add_argument('--small_validation', dest='small_validation',
                        default=0, type=int)

    parser.add_argument('--DP_K', dest='dp_k',
                        default=1.0, type=float)

    parser.add_argument('--DP_ETA', dest='eta_const',
                        default=-1.38, type=float)

    parser.add_argument('--dropout_distribution', dest='dropout_distribution',
                       default='bernoulli',
                       type=str)

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=['small', 'large'],
                      help='{small, large}',
                      default='small', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size during training',
                      type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                      help='max training epoch',
                      type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      type=bool)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      help='resume training',
                      type=bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                      help='vqav2 dataset root path',
                      type=str)

    parser.add_argument('--FEAT_PATH', dest='FEATURE_PATH',
                      help='bottom up features root path',
                      type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ###change wd to where the python file is
    import os

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    HP = Cfgs()

    args = parse_args()
    args_dict = HP.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = {**yaml_dict, **args_dict}
    HP.add_args(args_dict)
    HP.proc()

    print('Hyper Parameters:')
    print(HP)

    HP.check_path()

    execution = Execution(HP)
    execution.run(HP.RUN_MODE)




