

# imports
import sqlite3

import dcc.dcc_utils as dutils 


# constants
logger = dutils.get_logger(__name__)
SQL_SELECT_ALL_GENES = "select distinct gene_name, gene_ontology_id from nmf_ontology_gene"
SQL_SELECT_GENE_NAMES_FROM_LIST = "select distinct gene_name from nmf_ontology_gene where gene_synonym in ({})"
SQL_SELECT_GENE_CURIES_FROM_LIST = "select distinct gene_ontology_id from nmf_ontology_gene where gene_synonym in ({})"

# methods
def db_sqlite_get_connection(db_path):
    ''' 
    get the db connection 
    '''
    conn = sqlite3.connect(db_path)

    # return
    return conn


def db_get_all_genes(conn, log=True):
    '''
    returns a list of gene names in the DB
    '''
    list_genes = []
    cursor = conn.cursor()

    # get the list
    cursor.execute(SQL_SELECT_ALL_GENES)
    db_results = cursor.fetchall()
    list_genes = [{'gene_name': row[0], 'gene_ontology_id': row[1]} for row in db_results]

    # log
    if log:
        logger.info("Got list of size: {} DB distinct genes".format(len(list_genes)))

    # return
    return list_genes


def db_get_gene_names_from_list(conn, list_input, log=True):
    '''
    returns the genes names based on the input list of sterings
    '''
    list_genes = []
    cursor = conn.cursor()

    # build the query
    placeholders = ', '.join('?' for _ in list_input)
    sql_query = SQL_SELECT_GENE_NAMES_FROM_LIST.format(placeholders)

    # query
    cursor.execute(sql_query, list_input)
    db_results = cursor.fetchall()

    # build the results
    list_genes = [row[0] for row in db_results]

    # log
    if log:
        logger.info("for input list of size: {}, returning gene name list of size: {}".format(len(list_input), len(list_genes)))

    # return
    return list_genes

def db_get_gene_curies_from_list(conn, list_input, log=True):
    '''
    returns the genes curies based on the input list of sterings
    '''
    list_genes = []
    cursor = conn.cursor()

    # build the query
    placeholders = ', '.join('?' for _ in list_input)
    sql_query = SQL_SELECT_GENE_CURIES_FROM_LIST.format(placeholders)

    # query
    cursor.execute(sql_query, list_input)
    db_results = cursor.fetchall()

    # build the results
    list_genes = [row[0] for row in db_results]

    # log
    if log:
        logger.info("for input list of size: {}, returning gene curie list of size: {}".format(len(list_input), len(list_genes)))

    # return
    return list_genes


def db_load_gene_table_into_map(conn, log=False):
    '''
    will read the gene database and return a map of gene to position in array, map of gene to ontology_id
    '''
    # initialize
    map_gene_to_array_pos = {}
    list_temp = []
    map_gene_to_ontology_id = {}

    # read in the data
    list_temp = db_get_all_genes(conn=conn)
    for row in list_temp:
        map_gene_to_ontology_id[row.get('gene_name')] = row.get('gene_ontology_id')

    # create the map from the list; make sure no duplicates
    list_unique_gene = list(set(map_gene_to_ontology_id.keys()))
    list_unique_gene.sort()
    map_gene_to_array_pos = {value: index for index, value in enumerate(list_unique_gene)}

    # log
    logger.info("loaded genes  num count map: {}".format(len(map_gene_to_array_pos)))

    # return
    return map_gene_to_array_pos, list_unique_gene, map_gene_to_ontology_id


def db_get_ontology_list(conn, list_input, log=True):
    '''
    
    '''
    pass


# main
if __name__ == "__main__":
    pass