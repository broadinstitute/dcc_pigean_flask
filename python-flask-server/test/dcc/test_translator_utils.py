
# imports
import pytest 
import json 
import logging

import dcc.translator_utils as tran_utils


# constants
# path_files = "python-flask-server/test/dcc/test_medium/"
# list_gene_set_file = ["gene_set_list_msigdb_c2.txt"]
file_trapi_json = "python-flask-server/test/data/3a7a6b9c-7bf9-4de2-89a6-4b67f350f452.json"
file_trapi_json = "python-flask-server/test/data/alzheimer.json"
file_trapi_json = "python-flask-server/test/data/dili.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# test methods
def test_get_genes_from_trapi():
    '''
    test parsing the geen names out of the trapi result json
    '''
    print("\n=============== IN test_get_genes_from_trapi()")
    # initialize
    map_result = None

    # read the json from the test file
    json_test = {}
    with open(file_trapi_json, "r") as f:
        json_test = json.load(f) 

    # get the gene list
    map_result = tran_utils.get_genes_from_trapi(json_trapi_result=json_test)

    # test
    assert map_result is not None
    assert len(map_result) > 4

    # print
    print(json.dumps(list(map_result.values()), indent=2))


