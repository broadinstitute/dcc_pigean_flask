

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
    map_gene_index, list_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

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

def test_generate_gene_vector_from_list():
    '''
    test creating the vector from the input genes 
    '''
    # initialize
    vector_result = None
    map_gene_index = None 
    list_gene = ["ACE", "SLC30A8", "PPARG", "TUBB2A"]

    # get the gene index
    map_gene_index, list_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the vector
    vector_result, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_gene, map_gene_index=map_gene_index, log=True)

    # assert
    assert map_gene_index is not None
    assert vector_result is not None

    # print
    print(list_gene)
    print(vector_result)


def test_sum_of_gene_row():
    '''
    tests sum_of_gene_row() method
    '''
    # TODO
    # data = np.array([1, 2, 3, 4, 5, 6])
    # row_indices = np.array([0, 0, 1, 2, 2, 3])
    # col_indices = np.array([0, 2, 2, 0, 1, 4])

    # # Create a CSR (Compressed Sparse Row) matrix
    # sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(4, 5))

    # # Display the sparse matrix
    # print("Sparse Matrix:\n", sparse_matrix.toarray())


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


