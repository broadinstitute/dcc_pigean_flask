
# imports
import pytest 
import json 
import logging


import dcc.file_utils as futils
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 


# constants
path_files = "python-flask-server/test/dcc/test_small/"
list_gene_set_file = ["gene_set_small01_test.txt", "gene_set_small02_test.txt"]
gene_file = "gene_small_test.txt"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# test methods
def test_calc_X_shift_scale():
    '''
    test computing the mean shifts and scale factors
    '''
    print("\n=============== IN test_calc_X_shift_scale()")
    # initialize
    matrix_result = None
    map_gene_set_index = None
    map_gene_index = None 

    # get the gene index
    map_gene_index = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the data
    matrix_result, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # test the mean factors generation
    print("for test_calc_X_shift_scale, got matrix shape: {}".format(matrix_result.shape))
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_result)

    # test
    assert mean_shifts is not None
    print("for test_calc_X_shift_scale, got mean_shifts shape: {} and data: {}".format(mean_shifts.shape, mean_shifts))
    assert scale_factors is not None
    print("for test_calc_X_shift_scale, got scale_factors shape: {} and data: {}".format(scale_factors.shape, scale_factors))



def test_compute_beta_tildes():
    '''
    test computing the gene set matrix and gene vector
    '''
    print("\n=============== IN test_compute_beta_tildes()")
    # initialize
    matrix_gene_sets = None
    vector_genes = None
    map_gene_set_index = None
    map_gene_index = None 
    list_gene = ["ACE", "SLC30A8", "PPARG", "TUBB2A"]
    vector_pvalues = None

    # get the gene index
    map_gene_index = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the gene set data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # get the gene data
    vector_genes = mutils.generate_gene_vector_from_list(list_gene=list_gene, map_gene_index=map_gene_index, log=False)

    # test the mean factors generation
    print("for test_calc_X_shift_scale, got matrix shape: {}".format(matrix_gene_sets.shape))
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

    # test
    assert mean_shifts is not None
    assert scale_factors is not None

    # get the pvalues
    vector_pvalues = cutils.compute_beta_tildes(X=matrix_gene_sets, Y=vector_genes, scale_factors=scale_factors, mean_shifts=mean_shifts)

    # test 
    assert vector_pvalues is not None
    print("got pValues of shape: {} and data: {}".format(vector_pvalues.shape, vector_pvalues))
    # print("got pValues data: {}".format(vector_pvalues))




# def test_generate_gene_vector_from_list():
#     '''
#     test creating the vector from the input genes 
#     '''
#     # initialize
#     vector_result = None
#     map_gene_index = None 
#     list_gene = ["ACE", "SLC30A8", "PPARG", "TUBB2A"]

#     # get the gene index
#     map_gene_index = futils.load_gene_file_into_map(file_path=path_files + gene_file)

#     # get the vector
#     vector_result = mutils.generate_gene_vector_from_list(list_gene=list_gene, map_gene_index=map_gene_index, log=True)

#     # assert
#     assert map_gene_index is not None
#     assert vector_result is not None

#     # print
#     print(list_gene)
#     print(vector_result)




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


