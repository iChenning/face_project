import os
import logging
import sys

class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(logger, folder_dir):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler_file = logging.FileHandler(os.path.join(folder_dir, "training.log"))
    handler_file.setFormatter(formatter)
    handler_file.setLevel(level=logging.INFO)
    logger.addHandler(handler_file)

    handler_stream = logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(level=logging.INFO)
    logger.addHandler(handler_stream)

    logger.propagate = 0

