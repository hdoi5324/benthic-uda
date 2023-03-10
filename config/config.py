import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
#
__C.DATASET = edict()
__C.DATASET.NUM_CLASSES = 0
__C.DATASET.DATASET = ''
__C.DATASET.DATAROOT = ''
__C.DATASET.SOURCE_NAME = ''
__C.DATASET.TARGET_NAME = ''
__C.DATASET.VAL_NAME = ''

# Model options
__C.MODEL = edict()
__C.MODEL.FEATURE_EXTRACTOR = 'resnet101'
__C.MODEL.PRETRAINED = True
__C.MODEL.BP = False

# Bilinear Pooling options
__C.BP = edict()
__C.BP.AB = 64
__C.BP.R = 16
__C.BP.Q = 4

# __C.MODEL.FC_HIDDEN_DIMS = ()  ### options for multiple layers.

# data pre-processing options
#
__C.DATA_TRANSFORM = edict()
__C.DATA_TRANSFORM.TYPE = 'ours'  ### trun to simple for the VisDA dataset.

# Testing options
#
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 64

# Training options
#
__C.TRAIN = edict()
# batch size setting
__C.TRAIN.SOURCE_BATCH_SIZE = 32
__C.TRAIN.TARGET_BATCH_SIZE = 32

# learning rate schedule
__C.TRAIN.BASE_LR = 0.02
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.OPTIMIZER = 'SGD'
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LR_SCHEDULE = 'inv'
__C.TRAIN.MAX_EPOCH = 49
__C.TRAIN.SAVING = False   ## whether to save the intermediate status of model.
__C.TRAIN.PROCESS_COUNTER = 'iteration'
# __C.TRAIN.STOP_THRESHOLDS = (0.001, 0.001, 0.001)
# __C.TRAIN.TEST_INTERVAL = 1.0 # percentage of total iterations each loop
# __C.TRAIN.SAVE_CKPT_INTERVAL = 1.0 # percentage of total iterations in each loop

# Neptune Logging options
__C.NEPTUNE = edict()
__C.NEPTUNE_LOGGING = False
__C.NEPTUNE.API_TOKEN = 'xxxx' # Neptune API tokin
__C.NEPTUNE.PROJECT = 'userid/project_name' # Neptune username and project code

__C.STRENGTHEN = edict()
__C.STRENGTHEN.DATALOAD = 'normal'  ## normal | hard | soft. The original class aware sampling adopt the hard mode.
__C.STRENGTHEN.PERCATE = 10
__C.STRENGTHEN.CLUSTER_FREQ = 6



# optimizer options
__C.MCDALNET = edict()
__C.MCDALNET.DISTANCE_TYPE = ''  ## choose in L1 | KL | CE | MDD | DANN | SourceOnly

# optimizer options
__C.ADAM = edict()  ### adopted by the Digits dataset only
__C.ADAM.BETA1 = 0.9
__C.ADAM.BETA2 = 0.999

__C.INV = edict()
__C.INV.ALPHA = 10.0
__C.INV.BETA = 1.5 # 0.75

__C.OPEN = edict()
__C.OPEN.WEIGHT_UNK = 6.0



# MISC
__C.RESUME = ''
__C.TASK = 'closed'  ## closed | partial | open
__C.EVAL_METRIC = "accu"  # "mean_accu" as alternative
__C.EXP_NAME = 'exp'
__C.SAVE_DIR = ''
__C.NUM_WORKERS = 12
__C.PRINT_STEP = 3
__C.SEED = 55

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k in a:
        # a must specify keys that are in b
        v = a[k]
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    if filename[-1] == '\r':
        filename = filename[:-1]  ## delete the  '\r' at the end of the str
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value