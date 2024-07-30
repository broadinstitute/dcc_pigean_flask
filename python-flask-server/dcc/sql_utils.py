

# imports
import sqlite3

import dcc.dcc_utils as dutils 


# constants
logger = dutils.get_logger(__name__)
SQL_SELECT_ALL_GENES = "select distinct gene_name from nmf_ontology_gene"
SQL_SELECT_GENE_NAMES_FROM_LIST = "select distinct gene_name from nmf_ontology_gene where gene_synonym in ({})"

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
    list_genes = [row[0] for row in db_results]

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

def db_get_ontology_list(conn, list_input, log=True):
    '''
    
    '''
    pass


# main
if __name__ == "__main__":
    pass