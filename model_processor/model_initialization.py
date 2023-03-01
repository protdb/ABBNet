import os

import torch
from config.config import BaseConfig
from model.base_model import ABBModel
import logging


def load_shared_model():
    config = BaseConfig()
    model_path = config.get_model_path()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ABBModel().float()
    model.to(device)
    logger = logging.getLogger()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        logger.info(msg=f'Model loaded {model_path}')
    else:
        logger.error(msg=f'Model not found {model_path}')
        raise (FileNotFoundError, f'Model not found {model_path}')
    model.share_memory()
    return model
