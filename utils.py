import os, sys
import random
import logging
import numpy as np

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_logger(logdir):
    logger = logging.getLogger("default")
    cmd_handler = logging.StreamHandler(sys.stdout)
    cmd_handler.setLevel(logging.DEBUG)
    cmd_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    log_handler = logging.FileHandler(os.path.join(logdir, "train.log"), mode="w+", encoding="utf-8")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    logger.addHandler(cmd_handler)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    return logger
