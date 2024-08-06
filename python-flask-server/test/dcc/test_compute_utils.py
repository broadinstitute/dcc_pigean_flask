
# imports
import pytest 
import json 
import logging
import numpy as np

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
    map_gene_index, list_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

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
    map_gene_index, list_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the gene set data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # get the gene data
    vector_genes, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_gene, map_gene_index=map_gene_index, log=False)

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


def test_filter_matrix_columns():
    '''
    test the column filtering
    '''
    print("\n=============== IN test_filter_matrix_columns()")

    # initialize
    matrix01 = None
    matrix00 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
    ])
    cutoff = 0.05

    # Example p-values array
    p_values = np.array([0.01, 0.2, 0.03, 0.5])

    # get result
    matrix01, selected_column_indices = cutils.filter_matrix_columns(matrix_input=matrix00, vector_input=p_values, cutoff_input=cutoff, log=True)

    # test 
    assert matrix01 is not None
    print("got select columns: {}".format(selected_column_indices))


def test_filter_matrix_rows_by_sum_cutoff():
    '''
    test the row sum filtering
    '''
    print("\n=============== IN test_filter_matrix_rows_by_sum_cutoff()")
    # initialize
    matrix_result = None
    X = np.array([
        [1, -1, 0],
        [0, 1, 0],
        [2, 4, -5],
        [-1, -1, -1]
    ])

    V = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [100, 110, 120]
    ])

    # get the filtered matrix
    matrix_result, select_row_indices = cutils.filter_matrix_rows_by_sum_cutoff(matrix_to_filter=V, matrix_to_sum=X, log=True)

    # test
    assert matrix_result is not None
    print("got result matrix of shape: {} and data: \n{}".format(matrix_result.shape, matrix_result))
    print("got select rows: {}".format(select_row_indices))


def test_bayes_nmf_l2():
    '''
    test the bayes NMF
    '''
    print("\n=============== IN test_bayes_nmf_l2()")
    # initialize
    X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
    ])

    # get the result
    factor00, factor01, _, _, _, _ = cutils._bayes_nmf_l2(V0=X)

    # test
    assert np.all(factor00 >= 0)
    assert np.all(factor01 >= 0)
    print("got factor shapes: {} and {}".format(factor00.shape, factor01.shape))
    print("got factors: \n{} \n{}".format(factor00, factor01))



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


