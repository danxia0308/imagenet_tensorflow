import logging

def getLogger(log_file_path, level=logging.DEBUG, name="imagenet"):
    logger=logging.getLogger(name)
    logger.setLevel(level)
    fh=logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt=logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger