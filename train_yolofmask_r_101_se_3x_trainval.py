import sys
#sys.path.insert(0, './detectron2')

import argparse
import sys
import os
import torch, detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from detectron2.engine import default_setup

from tools.lazyconfig_train_net import do_train

dataset = 'coco2017'
annot_dir = './coco2017/annotations'
imgs_dir = './coco2017/{}2017'

for split in ['train', 'val']:
    annot_path = os.path.join(annot_dir, f'instances_{split}2017.json')
    d_name = dataset + f'_{split}'
    register_coco_instances(d_name, {}, annot_path, imgs_dir.format(split))

# Load dataset
dataset_dicts = DatasetCatalog.get('coco2017_train')
metadata = MetadataCatalog.get('coco2017_train')

config_file = "yolof_mask/configs/yolof_mask/yolofmask_r_101_se_3x.py"

class Args(argparse.Namespace):
    config_file=config_file
    eval_only=False
    num_gpus=1
    num_machines=1
    resume=False

args = Args()

cfg = LazyConfig.load(config_file)

default_setup(cfg, args)

do_train(args, cfg)