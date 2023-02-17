# -*- coding:utf-8 -*-
import os

from configs.cfg import Config
from dataloader import build_loader

import sys
# sys.path.insert(0, "../input/timm-efficientdet-pytorch")
# sys.path.insert(0, "../input/omegaconf")

import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
from  detector.head import BetterHead
from detector import Block
from utils.train import Fitter

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

def cfg(config_file):
    current_path = os.path.abspath(os.path.dirname(__file__))
    cfg = Config(current_path, config_file)
    return cfg
    # data_loader = build_loader(cfg, name)
    # return data_loader

# m = Block(5)
#
# input = torch.randn(16, 3, 32, 32)
# y = m(input)


if __name__ == '__main__':
    # for  j, k in enumerate(loader("flower.yaml", "train")):

    #     print(j, k)
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config = cfg("flower.yaml")
    model = Block(config.num_classes)
    fitter = Fitter(model, device, config)

    train_loader = build_loader(config,  "train")
    val_loader = build_loader(config,  "valid")
    fitter.fit(train_loader, val_loader)



