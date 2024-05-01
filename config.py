# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------\
_C.TOOLBOX_MODE = "only_test"
# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.OUTPUT_SAVE_DIR = ''
_C.TEST.METRICS = []
_C.TEST.USE_LAST_EPOCH = True
# Test.Data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.INFO = CN()
_C.TEST.DATA.INFO.LIGHT = ['']
_C.TEST.DATA.INFO.MOTION = ['']
_C.TEST.DATA.INFO.EXERCISE = [True]
_C.TEST.DATA.INFO.SKIN_COLOR = [1]
_C.TEST.DATA.INFO.GENDER = ['']
_C.TEST.DATA.INFO.GLASSER = [True]
_C.TEST.DATA.INFO.HAIR_COVER = [True]
_C.TEST.DATA.INFO.MAKEUP = [True]
_C.TEST.DATA.FILTERING = CN()
_C.TEST.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.TEST.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.TEST.DATA.FILTERING.SELECT_TASKS = False
_C.TEST.DATA.FILTERING.TASK_LIST = ['']
_C.TEST.DATA.FS = 0
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
_C.TEST.DATA.FILE_LIST_PATH = os.path.join(_C.TEST.DATA.CACHED_PATH, 'DataFileLists')
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DO_PREPROCESS = False
_C.TEST.DATA.DATA_FORMAT = 'NDCHW'
_C.TEST.DATA.BEGIN = 0.0
_C.TEST.DATA.END = 1.0
_C.TEST.DATA.FOLD = CN()
_C.TEST.DATA.FOLD.FOLD_NAME = ''
_C.TEST.DATA.FOLD.FOLD_PATH = ''
# Test Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.DATA_AUG = ['None']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TEST.DATA.PREPROCESS.CROP_FACE = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.TEST.DATA.PREPROCESS.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.RESIZE.W = 128
_C.TEST.DATA.PREPROCESS.RESIZE.H = 128
_C.TEST.DATA.PREPROCESS.BIGSMALL = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9

### -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_DIR = 'PreTrainedModels'

# Specific parameters for physnet parameters
_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 64

# -----------------------------------------------------------------------------
# Specific parameters for iBVPNet parameters
# -----------------------------------------------------------------------------
_C.MODEL.iBVPNet = CN()
_C.MODEL.iBVPNet.FRAME_NUM = 64

# -----------------------------------------------------------------------------
# Model Settings for TS-CAN
# -----------------------------------------------------------------------------
_C.MODEL.TSCAN = CN()
_C.MODEL.TSCAN.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Settings for EfficientPhys
# -----------------------------------------------------------------------------
_C.MODEL.EFFICIENTPHYS = CN()
_C.MODEL.EFFICIENTPHYS.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Settings for BigSmall
# -----------------------------------------------------------------------------
_C.MODEL.BIGSMALL = CN()
_C.MODEL.BIGSMALL.FRAME_DEPTH = 3

# -----------------------------------------------------------------------------
# Model Settings for PhysFormer
# -----------------------------------------------------------------------------
_C.MODEL.PHYSFORMER = CN()
_C.MODEL.PHYSFORMER.PATCH_SIZE = 4
_C.MODEL.PHYSFORMER.DIM = 96
_C.MODEL.PHYSFORMER.FF_DIM = 144
_C.MODEL.PHYSFORMER.NUM_HEADS = 4
_C.MODEL.PHYSFORMER.NUM_LAYERS = 12
_C.MODEL.PHYSFORMER.THETA = 0.7

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
_C.INFERENCE.EVALUATION_WINDOW = CN()
_C.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
_C.INFERENCE.MODEL_PATH = './final_model_release/PURE_DeepPhys.pth'

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cpu"
_C.NUM_OF_GPU_TRAIN = 1

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> Merging a config file from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):

    # store default file list path for checking against later
    default_TEST_FILE_LIST_PATH = config.TEST.DATA.FILE_LIST_PATH

    # update flag from config file
    _update_config_from_file(config, args.config_file)
    config.defrost()
    
    # UPDATE TRAIN PATHS


    # UPDATE TEST PATHS
    if config.TEST.DATA.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, 'DataFileLists')

    if config.TEST.DATA.EXP_DATA_NAME == '':
        config.TEST.DATA.EXP_DATA_NAME = "_".join([config.TEST.DATA.DATASET, "SizeW{0}".format(
            str(config.TEST.DATA.PREPROCESS.RESIZE.W)), "SizeH{0}".format(str(config.TEST.DATA.PREPROCESS.RESIZE.H)), "ClipLength{0}".format(
            str(config.TEST.DATA.PREPROCESS.CHUNK_LENGTH)), "DataType{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_TYPE)),
                                      "DataAug{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_AUG)),
                                      "LabelType{0}".format(config.TEST.DATA.PREPROCESS.LABEL_TYPE),
                                      "Crop_face{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                      "Backend{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                      "Large_box{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                      "Large_size{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                      "Dyamic_Det{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                        "det_len{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                        "Median_face_box{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                              ])
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    name, ext = os.path.splitext(config.TEST.DATA.FILE_LIST_PATH)
    if not ext: # no file extension
        FOLD_STR = '_' + config.TEST.DATA.FOLD.FOLD_NAME if config.TEST.DATA.FOLD.FOLD_NAME else ''
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.FILE_LIST_PATH, \
                                                       config.TEST.DATA.EXP_DATA_NAME + '_' + \
                                                       str(config.TEST.DATA.BEGIN) + '_' + \
                                                       str(config.TEST.DATA.END) + \
                                                       FOLD_STR + '.csv')
    elif ext != '.csv':
        raise ValueError('TEST dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    if ext == '.csv' and config.TEST.DATA.DO_PREPROCESS:
        raise ValueError('User specified TEST dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TEST dataset FILE_LIST_PATH .csv file.')
    

    config.freeze()
    return

def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


