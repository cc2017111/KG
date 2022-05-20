import argparse
from pathlib import Path
import os
import tensorflow as tf
from configure import Configure
from utils.logger import get_logger
from data import BertDataManager
from train import train


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
base_path = Path(__file__).resolve().parent.parent


def check_fold(configures):
    datasets_fold = 'datasets_fold'
    assert hasattr(configures, datasets_fold), 'item datasets_fold not configured'
    print(os.path.join(base_path, configures.datasets_fold))
    if not os.path.exists(os.path.join(base_path, configures.datasets_fold)):
        print('datasets fold not found')
        exit(1)

    checkpoint_dir = 'checkpoint_dir'
    if not hasattr(configures, checkpoint_dir) or not os.path.exists(os.path.join(base_path, configures.checkpoint_dir)):
        print('checkpoints fold not found, creating...')
        paths = configures.checkpoint_dir.split('/')
        if len(paths) == 2 and os.path.exists(paths[0]) and not os.path.exists(os.path.join(base_path, configures.checkpoint_dir)):
            os.mkdir(os.path.join(base_path, configures.checkpoint_dir))
        else:
            os.mkdir(os.path.join(base_path, 'checkpoints'))

    vocabs_dir = 'vocabs_dir'
    if not os.path.exists(os.path.join(base_path, configures.vocabs_dir)):
        print('vocabs fold not found, creating...')
        if hasattr(configures, vocabs_dir):
            os.mkdir(os.path.join(base_path, configures.vocabs_dir))
        else:
            os.mkdir(os.path.join(base_path, configures.datasets_fold + '/vocabs'))

    log_dir = 'log_dir'
    if not os.path.exists(os.path.join(base_path, configures.log_dir)):
        print('log fold not found, creating...')
        if hasattr(configures, log_dir):
            os.mkdir(os.path.join(base_path, configures.log_dir))
        else:
            os.mkdir(os.path.join(base_path, configures.datasets_fold + '/logs'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert_TextCNN')
    parser.add_argument('--config_file', default='Bert_textCNN.config')
    args = parser.parse_args()
    configs = Configure(config_file=os.path.join(base_path, args.config_file))
    check_fold(configs)
    logger = get_logger(configs.log_dir)
    mode = configs.mode.lower()
    dataManager = BertDataManager(configs, logger)
    if mode == "train":
        logger.info("mode:train")
        train(configs, dataManager, logger)
