# -*- coding: utf-8 -*-
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re


class Logger(object):
    def __init__(self, path, clevel=logging.WARNING, flevel=logging.INFO):
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        if self.logger.handlers:
            return
        # set cmd log
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # set the file log
        # path = 'riskmodel/log' + os.sep + 'log'
        # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log', 'log')
        fh = TimedRotatingFileHandler(path, when='midnight', backupCount=4)
        fh.suffix = '%Y%m%d.log'
        fh.extMatch = re.compile(r"^\d{8}.log$")
        # fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == '__main__':
    log_risk = Logger(logging.INFO, logging.DEBUG)
    log_risk.debug('debug')
    log_risk.warn('warning')
    log_risk.error('error')
    log_risk.info('info')
    log_risk.info('critical')
