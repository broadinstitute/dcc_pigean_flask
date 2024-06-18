
# imports
import logging
import sys 

# settings
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s %(threadName)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)


# methods
def get_logger(name): 
    # get the logger
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    # return
    return logger 
