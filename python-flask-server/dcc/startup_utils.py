

# imports
import json
from dcc_utils import get_logger

# constants
DIR_CONF = "python-flask-server/conf/"
FILE_CONF = "startup_files.json"

# variables
matrix_gene_set = None
vector_genes = None
map_conf = None
logger = get_logger(__name__)

# methods
def load_conf(dir=DIR_CONF, log=True):
    '''
    will load the configuration
    '''
    file_conf = dir + FILE_CONF
    json_data = None

    # log
    if log:
        logger.info("loading startup conf file: {}".format(file_conf))

    # load the map
    with open(file_conf) as f:
         json_data = json.load(f)

    # return
    return json_data

def get_conf():
    '''
    will return the configuration
    '''
    if not map_conf:
        map_conf = load_conf()

    # return
    return map_conf


# main
if __name__ == "__main__":
    pass
