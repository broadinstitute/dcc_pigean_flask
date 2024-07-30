
# imports
import pytest 
import json 

import dcc.sql_utils as sutils

# constants
db_file = "python-flask-server/conf/startup_files/pigean.db"

# test methods
def test_db_get_all_genes():
    '''
    test getting all the genes from the DB
    '''
    # initialize
    list_result = []

    # get the db connection
    conn = sutils.db_sqlite_get_connection(db_path=db_file)

    # get the data
    list_result = sutils.db_get_all_genes(conn=conn)

    # test
    assert len(list_result) == 19500


def test_db_get_gene_names_from_list():
    '''
    test getting the gene names given a list
    '''
    # initialize
    list_result = []
    list_input = ['ACE', 'TRAPPC2B', 'NCBIGene:10597', 'NCBIGene:23380', 'LINC02203']

    # get the db connection
    conn = sutils.db_sqlite_get_connection(db_path=db_file)

    # get the data
    list_result = sutils.db_get_gene_names_from_list(conn=conn, list_input=list_input)

    # test
    assert len(list_input) == 5
    assert len(list_result) == 4


def test_db_load_gene_table_into_map():
    '''
    test loading the genes data from the db
    '''
    # initialize
    map_gene_to_array_pos = {}
    list_genes = []
    map_gene_to_ontology_id = {}

    # get the db connection
    conn = sutils.db_sqlite_get_connection(db_path=db_file)

    # get the data
    map_gene_to_array_pos, list_genes, map_gene_to_ontology_id = sutils.db_load_gene_table_into_map(conn=conn)

    # test
    assert len(list_genes) == 19500
    assert len(map_gene_to_array_pos) == 19500
    assert len(map_gene_to_ontology_id) == 19500

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
