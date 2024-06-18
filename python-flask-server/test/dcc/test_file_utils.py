
# imports
import pytest 
import json 

import dcc.file_utils as futils

# constants
gene_set_file = "python-flask-server/test/dcc/gene_set_list_test.txt"
gene_file = "python-flask-server/test/dcc/gene_test.txt"

# test methods
def test_read_tab_delimited_file():
    '''
    test file reading into a map
    '''
    # initialize
    map_result = {}

    # get the data
    map_result = futils.read_tab_delimited_file(file_path=gene_set_file)

    # test
    assert len(map_result) == 8
    assert map_result.get('HALLMARK_UNFOLDED_PROTEIN_RESPONSE') is not None
    assert len(map_result.get('HALLMARK_UNFOLDED_PROTEIN_RESPONSE')) == 17


def test_load_gene_file_into_map():
    '''
    test file reading into a map
    '''
    # initialize
    map_result = {}

    # get the data
    map_result = futils.load_gene_file_into_map(file_path=gene_file)

    # test
    assert len(map_result) == 200
    assert map_result.get('CXCL10') is not None
    # assert map_result.get('CXCL10') == 56


# def test_get_querygraph_key_node():
#     '''
#     test the qnode retrieval
#     '''
#     # initialize
#     query: Query = Query.from_dict(json_data)
#     key = None 
#     qnode: QNode = None

#     # get the data
#     key, qnode = textract.get_querygraph_key_node(trapi_query=query)

#     # test
#     assert key == "subj"
#     assert qnode.ids == ["MONDO:0011936"]
#     assert qnode.set_interpretation == "BATCH"

#     # get the data
#     key, qnode = textract.get_querygraph_key_node(trapi_query=query, is_subject=False)

#     # test
#     assert key == "obj"
#     assert qnode.ids is None
#     assert qnode.set_interpretation is None
