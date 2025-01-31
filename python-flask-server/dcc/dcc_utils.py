
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
import logging
import sys 

# settings
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s %(threadName)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)

# constants
VERSION_CODE = "20240904_yvar_fix"
NUMBER_RETURNED_PER_FACTOR=5

# startup keys
KEY_DEFAULT_GENE_SET_FAMILY = "default"
KEY_GENE_SET_FILES = "gene_set_files"
KEY_FILE_ROOT_DIR = "root_dir"
KEY_GENE_SET_FAMILIES = "gene_set_families"
KEY_NAME_GENE_SET_FAMILY = "name"

# app keys
KEY_REST_GENES = "genes"
KEY_REST_P_VALUE = "p_value"
KEY_REST_MAX_NUMBER_GENE_SETS = "max_number_gene_sets"
KEY_REST_GENE_SET = "gene_sets"
KEY_INTERNAL_LOWEST_FACTOR_SCORE = "lowest_factor_score"
KEY_INTERNAL_HIGHEST_FACTOR_NAME = "highest_factor_name"
KEY_INTERNAL_HIGHEST_FACTOR_SCORE = "highest_factor_score"
KEY_REST_GENERATE_FACTOR_LABELS = "generate_factor_labels"
KEY_REST_ADD_GENE_SCORES = "calculate_gene_scores"

# app call keys
KEY_APP_QUERY=1
KEY_APP_NOVELTY=2
KEY_APP_PIGEAN=3

# pigean app keys
KEY_APP_FACTOR_PIGEAN = 'pigean-factor'
KEY_APP_FACTOR_GENE = 'gene-factor'
KEY_APP_FACTOR_GENE_SET = 'gene-set-factor'
KEY_APP_INPUT_GENES = 'input_genes'
KEY_APP_GENE_SETS = 'gene_sets'
KEY_APP_GENE_SET = 'gene_set'
KEY_APP_P_VALUE = 'p_value'
KEY_APP_GENE_SET_SCORES = 'gene_set_scores'
KEY_APP_GENE_SCORES = 'gene_scores'
KEY_APP_NETWORK_GRAPH = 'network_graph'


# methods
def get_logger(name): 
    # get the logger
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    # return
    return logger 

def get_code_version():
    '''
    returns the code version
    '''
    return VERSION_CODE



