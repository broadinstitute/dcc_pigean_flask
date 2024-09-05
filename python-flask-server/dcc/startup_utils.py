
# LICENSE
# Copyright 2024 Flannick Lab

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

# imports
import json
import dcc.dcc_utils as dutils 
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 

# constants
# DIR_CONF = "python-flask-server/conf/"
DIR_CONF = "conf/"
FILE_CONF = "startup_files.json"

# variables
# matrix_gene_set = None
# vector_genes = None
# map_conf = None
logger = dutils.get_logger(__name__)

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

def load_gene_set_family_map(map_conf, map_gene_index, log=False):
    '''
    method to load the gene set family data into the objects
    '''
    map_gene_set_family = {}
    dir_root = map_conf.get(dutils.KEY_FILE_ROOT_DIR)

    # load the default data
    map_gene_set_family[dutils.KEY_DEFAULT_GENE_SET_FAMILY] = load_gene_set_family_data(name=dutils.KEY_DEFAULT_GENE_SET_FAMILY, 
        list_gene_set_files=map_conf.get(dutils.KEY_GENE_SET_FILES), path_gene_set_files=dir_root, map_gene_index=map_gene_index, log=log)

    # load the extra data
    # make sure there is a gene set family value
    list_gene_set_families_config = map_conf.get(dutils.KEY_GENE_SET_FAMILIES)
    if list_gene_set_families_config:
        for item in list_gene_set_families_config:
            name = item.get(dutils.KEY_NAME_GENE_SET_FAMILY)
            map_gene_set_family[name] = load_gene_set_family_data(name=name, list_gene_set_files=item.get(dutils.KEY_GENE_SET_FILES),
            path_gene_set_files=dir_root, map_gene_index=map_gene_index, log=log)

    # log
    logger.info("loaded gene set family list of size: {} with name: {}".format(len(map_gene_set_family), list(map_gene_set_family.keys())))

    # return
    return map_gene_set_family


def load_gene_set_family_data(name, list_gene_set_files, path_gene_set_files, map_gene_index, log=False):
    '''
    load a gene set family data structure 
    '''
    # initialize
    class_gene_set_family = GeneSetFamily(name=name, list_gene_set_files=list_gene_set_files, path_gene_set_files=path_gene_set_files)

    # log
    logger.info("loading gene set family: {}".format(name))

    # load matrix data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, 
                                                                  list_gene_set_files=list_gene_set_files, path_gene_set_files=path_gene_set_files, log=log)
    class_gene_set_family.matrix_gene_sets = matrix_gene_sets
    class_gene_set_family.map_gene_set_index = map_gene_set_index
    # TODO - fix
    class_gene_set_family.list_gene_sets = []

    # load the calculated data
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)
    class_gene_set_family.mean_shifts = mean_shifts
    class_gene_set_family.scale_factors = scale_factors

    # log
    logger.info("loading gene set family: {} with gene set matrix of shape: {}".format(name, matrix_gene_sets.shape))

    # return
    return class_gene_set_family


# classes
class GeneSetFamily:
    def __init__(self, name, list_gene_set_files, path_gene_set_files, matrix_gene_sets=None, map_gene_set_index=None, list_gene_sets=None, mean_shifts=None, scale_factors= None):
        self.name = name
        self.list_gene_set_files = list_gene_set_files
        self.path_gene_set_files = path_gene_set_files
        self.matrix_gene_sets = matrix_gene_sets
        self.map_gene_set_index = map_gene_set_index
        self.list_gene_sets = list_gene_sets
        self.mean_shifts = mean_shifts
        self.scale_factors = scale_factors


# main
if __name__ == "__main__":
    pass
