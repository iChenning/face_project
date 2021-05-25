import os
import logging
import sys


def logger_init(folder_dir):
    logger = logging.getLogger('run')
    logger.setLevel(level=logging.INFO)
    # streamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # fileHandler
    file_path = os.path.join(folder_dir, 'log.log')
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = 0
    return logger
