
# imports
import pytest 
import json 

import dcc.file_utils as futils
import dcc.matrix_utils as mutils 


# constants
path_files = "python-flask-server/test/dcc/test_small/"
list_gene_set_file = ["gene_set_small01_test.txt", "gene_set_small02_test.txt"]
gene_file = "gene_small_test.txt"

# test methods
def test_load_geneset_matrix():
    '''
    test loading the sparse matrix for calculation
    '''
    # initialize
    matrix_result = None
    map_gene_set_index = None
    map_gene_index = None 

    # get the gene index
    map_gene_index = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the data
    matrix_result, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=True)

    # test
    assert map_gene_index is not None
    assert matrix_result is not None
    assert map_gene_set_index is not None
    assert len(map_gene_index) == 10
    assert len(map_gene_set_index) == 3

    # print
    print(matrix_result)

    # print the maps
    print("genes: {}".format(json.dumps(map_gene_index, indent=2)))
    print("gene sets: {}".format(json.dumps(map_gene_set_index, indent=2)))


# def test_load_gene_file_into_map():
#     '''
#     test file reading into a map
#     '''
#     # initialize
#     map_result = {}

#     # get the data
#     map_result = futils.load_gene_file_into_map(file_path=gene_file)

#     # test
#     assert len(map_result) == 200
#     assert map_result.get('CXCL10') is not None
#     assert map_result.get('CXCL10') == 56


