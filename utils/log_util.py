"""
@author: edenmyn
@email: edenmyn
@time: 2022/6/23 9:06
@DESC: 

"""
import logging

format_str = '%(asctime)s %(levelname)s %(filename)s-%(lineno)d %(message)s'

logging.basicConfig(format=format_str)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
