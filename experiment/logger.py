# -*- coding: utf-8 -*-
__author__ = 'bk'

import logging

LOG_FILE_PATH = '../log/'
LOG_FILE_NAME = 'results'


def get_logger(name, log_file_name=LOG_FILE_NAME, level=logging.DEBUG):
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s] [%(name)s] %(message)s")
    root_logger = logging.getLogger(name=name)
    root_logger.setLevel(level=level)

    file_handler = logging.FileHandler("{0}/{1}.log".format(LOG_FILE_PATH, log_file_name))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    return root_logger
