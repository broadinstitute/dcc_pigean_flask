

# imports
import os
import pymysql as mdb
import sqlite3


# constants
DB_PASSWD = os.environ.get('DB_PASSWD')
DB_SCHEMA = "tran_upkeep"
DB_FILE_SQLITE = "/home/javaprog/Data/Broad/Translator/BayesGenSetNMF/Sqlite/pigean.db"

SQL_MYSQL_SELECT_GENES = "select node_code as gene_name, ontology_id as gene_synonym from {}.agg_gene"

SQL_SQLITE_DELETE_GENES = "delete from nmf_ontology_gene"
SQL_SQLITE_INSERT_GENES = "insert into nmf_ontology_gene (gene_synonym, gene_name) values(:gene_synonym, :gene_name)"


# methods
def db_mysql_get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=DB_SCHEMA)

    # return
    return conn


def db_sqlite_get_connection(db_path):
    ''' 
    get the db connection 
    '''
    conn = sqlite3.connect(db_path)

    # return
    return conn


def db_select_mysql_gene(conn, log=False):
    '''
    select the reference genes from the mysql upkeep db
    '''
    list_result = []

    # get the data
    cursor = conn.cursor()
    cursor.execute(SQL_MYSQL_SELECT_GENES.format(DB_SCHEMA))
    db_results = cursor.fetchall()

    # populate the list
    for row in db_results:
        list_result.append({'gene_name': row[0], 'gene_synonym': row[1]})

    # NEED TO USE dICTcURSOR FOR THIS TO WORK
    # list_result = [dict(row) for row in db_results]
        
    # return
    return list_result


def db_insert_sqlite_gene(conn, list_genes, log=False):
    '''
    insert the genes into the sqlite db
    '''
    # initialize
    cursor = conn.cursor()

    # loop and insert
    for index, row in enumerate(list_genes):
        cursor.execute(SQL_SQLITE_INSERT_GENES, {'gene_synonym': row.get('gene_synonym'), 'gene_name': row.get('gene_name')})
        cursor.execute(SQL_SQLITE_INSERT_GENES, {'gene_synonym': row.get('gene_name'), 'gene_name': row.get('gene_name')})

        conn.commit()

        # log
        if log:
            if index % 500:
                print("{}/{} - insert row: {}".format(index, len(list_genes), row))


def db_delete_sqlite_gene(conn, log=False):
    '''
    delete the genes into the sqlite db
    '''
    # initialize
    cursor = conn.cursor()

    # loop and insert
    cursor.execute(SQL_SQLITE_DELETE_GENES)
    conn.commit()


# main
if __name__ == "__main__":
    # get the db connections
    conn_mysql = db_mysql_get_connection()
    conn_sqlite = db_sqlite_get_connection(DB_FILE_SQLITE)

    # get the genes from mysql
    list_genes = db_select_mysql_gene(conn=conn_mysql)
    print("got gene list of size: {}".format(len(list_genes)))

    # delete the sqlite genes
    db_delete_sqlite_gene(conn=conn_sqlite)

    # insert the sqlite genes
    db_insert_sqlite_gene(conn=conn_sqlite, list_genes=list_genes, log=True)
    print("inserted gene list of size: {}".format(len(list_genes)))



