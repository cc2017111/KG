# -*- coding: utf-8 -*-
# @Time : 2021/6/30 11:04 
# @Author : jinyuhe
# @Email : ai_lab@toec.com
# @File : logger.py 
# @Software: PyCharm
import datetime
import logging
from pathlib import Path
base_path = Path(__file__).resolve().parent.parent.parent


def get_logger(log_dir):
    log_file = str(base_path) + '/' + log_dir + '/' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return logger
