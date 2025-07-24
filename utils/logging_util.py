import logging
import os

# 日志打印函数
# 在分布式训练时只在主进程(rank=0)上打印日志
def Logger(log, content, level: int=logging.DEBUG):
    rank = int(os.environ.get("RANK", -1))
    if rank == -1 or rank == 0:
        if logging.DEBUG == level:
            log.debug(content)
        elif logging.INFO == level:
            log.info(content)
        elif logging.WARNING == level:
            log.warning(content)
        elif logging.ERROR == level:
            log.error(content)
        elif logging.CRITICAL == level:
            log.critical(content)
        else:
            log.debug(content)