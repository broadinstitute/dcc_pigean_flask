
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

import numpy as np
from scipy.sparse import csc_matrix, hstack

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
    mock_family_index = 0

    # load the default data
    map_gene_set_family[dutils.KEY_DEFAULT_GENE_SET_FAMILY] = load_gene_set_family_data(name=dutils.KEY_DEFAULT_GENE_SET_FAMILY, 
        list_gene_set_files=map_conf.get(dutils.KEY_GENE_SET_FILES), path_gene_set_files=dir_root, map_gene_index=map_gene_index, log=log)
    # create a family with controls
    family_with_controls = cretate_mock_gene_set_family(map_gene_set_family[dutils.KEY_DEFAULT_GENE_SET_FAMILY], mock_family_index)
    map_gene_set_family[family_with_controls.name] = family_with_controls
    mock_family_index += 1

    # load the extra data
    # make sure there is a gene set family value
    # TODO - for gene set families, cat FileNotFoundError and do not load family if error
    list_gene_set_families_config = map_conf.get(dutils.KEY_GENE_SET_FAMILIES)
    if list_gene_set_families_config:
        for item in list_gene_set_families_config:
            name = item.get(dutils.KEY_NAME_GENE_SET_FAMILY)
            map_gene_set_family[name] = load_gene_set_family_data(name=name, list_gene_set_files=item.get(dutils.KEY_GENE_SET_FILES),
                path_gene_set_files=dir_root, map_gene_index=map_gene_index, log=log)
            family_with_controls = cretate_mock_gene_set_family(map_gene_set_family[name], mock_family_index)
            map_gene_set_family[family_with_controls.name] = family_with_controls
            mock_family_index += 1

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


def cretate_mock_gene_set_family(src_gene_set_family, suffix):
    '''
    create family with mock gene sets
    This will add mock gene sets with the same number of gene sets as the source family
    '''
    name = src_gene_set_family.name + dutils.KEY_NEGATIVE_CONTROLS
    list_gene_set_files = src_gene_set_family.list_gene_set_files
    path_gene_set_files = src_gene_set_family.path_gene_set_files
    src_matrix_gene_sets = src_gene_set_family.matrix_gene_sets
    src_map_gene_set_index = src_gene_set_family.map_gene_set_index
    n_gene_sets = src_matrix_gene_sets.shape[1]
    mock_matrix_gene_sets, mock_map_gene_set_index = create_mock_gene_set_matrix(src_gene_set_family.matrix_gene_sets, suffix)
    matrix_gene_sets = hstack([src_matrix_gene_sets, mock_matrix_gene_sets], format='csc')
    map_gene_set_index = src_map_gene_set_index.copy()
    for (i, mock_name) in mock_map_gene_set_index.items():
        map_gene_set_index[i + n_gene_sets] = mock_name
    list_gene_sets = []
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

    mock_gene_set_family = GeneSetFamily(name=name, list_gene_set_files=list_gene_set_files, 
                                         path_gene_set_files=path_gene_set_files, 
                                         matrix_gene_sets=matrix_gene_sets, 
                                         map_gene_set_index=map_gene_set_index, 
                                         list_gene_sets=list_gene_sets, 
                                         mean_shifts=mean_shifts, 
                                         scale_factors=scale_factors)
    # log
    logger.info("created mock gene set family {} with {} gene sets and {} genes".format(name, matrix_gene_sets.shape[1], matrix_gene_sets.shape[0]))
    # return
    return mock_gene_set_family


def create_mock_gene_set_matrix(src_matrix_gene_sets, suffix=0):
    n_gene_sets = src_matrix_gene_sets.shape[1] if src_matrix_gene_sets is not None else 0
    n_all_genes = src_matrix_gene_sets.shape[0] if src_matrix_gene_sets is not None else 0
    logger.info("creating mock gene set family with {} gene sets and {} genes".format(n_gene_sets, n_all_genes))
    map_gene_set_index = {}

    list_row = []
    list_columns = []
    list_data = []
    for i in range(n_gene_sets):
        map_gene_set_index[i] = 'negative_control_' + str(i) + '_' + str(suffix)
        n_genes = src_matrix_gene_sets[:,i].count_nonzero()
        all_genes = [j for j in range(n_all_genes)]
        for j in range(n_genes):
            random_index = np.random.randint(n_all_genes-j)
            gene_index = all_genes[random_index]
            all_genes[random_index] = all_genes[n_all_genes-j-1]  # remove the gene from the list
            list_row.append(gene_index)
            list_columns.append(i)
            list_data.append(1)

    matrix_gene_sets = csc_matrix((list_data, (list_row, list_columns)), shape=(n_all_genes, n_gene_sets))

    for i in range(n_gene_sets):
        src_count = src_matrix_gene_sets[:,i].count_nonzero()
        mock_count = matrix_gene_sets[:,i].count_nonzero()
        if src_count != mock_count:
            logger.warning("source gene set {} has {} genes, but mock gene set has {}".format(i, src_count, mock_count))

    logger.info("created mock gene set family with {} gene sets and {} genes".format(n_gene_sets, n_all_genes))
    return matrix_gene_sets, map_gene_set_index


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

    def get_num_gene_sets(self):
        if self.map_gene_set_index:
            return len(self.map_gene_set_index)
        else:
            return -1

# main
if __name__ == "__main__":
    pass
