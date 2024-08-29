# imports
from flask import Flask, render_template, request, flash

import dcc.startup_utils as sutils
import dcc.file_utils as futils
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 
import dcc.dcc_utils as dutils 
import dcc.sql_utils as sql_utils
import dcc.gui_utils as gutils

import time 

# constants
DEBUG=False

# variables
app = Flask(__name__)
app.secret_key = "test_app_gpt"

logger = dutils.get_logger(__name__)
# p_value_cutoff = 0.3
P_VALUE_CUTOFF = 0.3
# p_value_cutoff = 0.05
MAX_NUMBER_GENE_SETS_FOR_COMPUTATION=100

# in memory compute variables
map_conf = sutils.load_conf()

# load the data
db_file = map_conf.get('root_dir') +  map_conf.get('db_file')
logger.info("loading database file: {}".format(db_file))
sql_connection = sql_utils.db_sqlite_get_connection(db_path=db_file)

# map_gene_index, list_system_genes = futils.load_gene_file_into_map(file_path=map_conf.get('root_dir') + map_conf.get('gene_file'))
map_gene_index, list_system_genes, map_gene_ontology = sql_utils.db_load_gene_table_into_map(conn=sql_connection)
matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, 
                                                                  list_gene_set_files=map_conf.get('gene_set_files'), path_gene_set_files=map_conf.get('root_dir'), log=False)
(mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

print("================ Bayes App is UP! ===========================")


@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    map_result = {'message': 'yes, I am up ;>'}

    return map_result


@app.route("/query", methods=["POST"])
def post_genes():
    # initialize 
    map_result = {}
    list_input_genes = []
    sql_conn_query = sql_utils.db_sqlite_get_connection(db_path=db_file)
    p_value_cutoff = P_VALUE_CUTOFF

    # get the input
    data = request.get_json()
    if data:
        list_input_genes = data.get('genes')

    logger.info("got request: {} with gene inputs: {}".format(request.method, list_input_genes))
    logger.info("got gene inputs of size: {}".format(len(list_input_genes)))

    # get the p_value
    p_value_cutoff = process_numeric_value(json_request=data, name='p_value', cutoff_default=P_VALUE_CUTOFF)
    logger.info("got using p_value: {}".format(p_value_cutoff))

    # get the max gene sets
    max_number_gene_sets = process_numeric_value(json_request=data, name='max_number_gene_sets', cutoff_default=MAX_NUMBER_GENE_SETS_FOR_COMPUTATION, is_float=False)
    logger.info("got using p_value: {}".format(p_value_cutoff))

    # translate the genes into what the system can handle
    list_input_translated = sql_utils.db_get_gene_names_from_list(conn=sql_conn_query, list_input=list_input_genes)
    logger.info("got translated gene inputs of size: {}".format(len(list_input_translated)))

    # add the genes to the result
    # map_result['input_genes'] = list_input_genes
    if DEBUG:
        map_result['conf'] = map_conf

    # time
    start = time.time()

    # compute
    list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_gene_novelty, logs_process = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, 
                                                                                                               p_value=p_value_cutoff,
                                                                                                               max_num_gene_sets=max_number_gene_sets,
                                                                                                               list_gene=list_input_translated, 
                                                                                                               list_system_genes=list_system_genes, 
                                                                                                               map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
                                                                                                               mean_shifts=mean_shifts, scale_factors=scale_factors,
                                                                                                               log=True)

    # time
    end = time.time()

    # format the data
    # map_factors = cutils.group_factor_results(list_factor=list_factor, list_factor_gene_sets=list_factor_gene_sets, list_factor_genes=list_factor_genes)
    # map_result['data'] = map_factors
    map_result = gutils.gui_build_results_map(list_factor=list_factor, list_factor_gene_sets=list_factor_gene_sets, list_factor_genes=list_factor_genes, 
                                              map_gene_ontology=map_gene_ontology, list_input_gene_names=list_input_translated, map_gene_index=map_gene_index,
                                              matrix_gene_sets=matrix_gene_sets, map_gene_novelty=map_gene_novelty)


    # add time
    str_message = "total elapsed time is: {}s".format(end-start)
    logs_process.append(str_message)
    map_result['logs'] = logs_process
    logger.info(str_message)

    # return
    return map_result


@app.route("/novelty_query", methods=["POST"])
def post_novelty_genes():
    '''
    will respond to the novelty score
    '''
    # initialize 
    map_result = {}
    list_input_genes = []
    p_value_cutoff = P_VALUE_CUTOFF

    # get the input
    data = request.get_json()
    if data:
        list_input_genes = data.get('genes')

    logger.info("got request: {} with gene inputs: {}".format(request.method, list_input_genes))
    logger.info("got gene inputs of size: {}".format(len(list_input_genes)))

    # get the p_value
    p_value_cutoff = process_numeric_value(json_request=data, name='p_value', cutoff_default=P_VALUE_CUTOFF)
    logger.info("got using p_value: {}".format(p_value_cutoff))

    # get the max gene sets
    max_number_gene_sets = process_numeric_value(json_request=data, name='max_number_gene_sets', cutoff_default=MAX_NUMBER_GENE_SETS_FOR_COMPUTATION)
    logger.info("got using p_value: {}".format(p_value_cutoff))

    # get the calculated data
    map_gene_novelty, list_input_translated = process_genes(list_input_genes=list_input_genes, p_value_cutoff=p_value_cutoff)

    # format the data
    map_result = gutils.gui_build_novelty_results_map(map_gene_ontology=map_gene_ontology, list_input_gene_names=list_input_translated, map_gene_index=map_gene_index,
                                              matrix_gene_sets=matrix_gene_sets, map_gene_novelty=map_gene_novelty)


    # return
    return map_result



@app.route("/curie_query", methods=["POST"])
def post_gene_curies():
    # initialize 
    list_input_genes = []
    sql_conn_query = sql_utils.db_sqlite_get_connection(db_path=db_file)

    # get the input
    data = request.get_json()
    if data:
        list_input_genes = data.get('genes')

    logger.info("got request: {} with gene inputs: {}".format(request.method, list_input_genes))
    logger.info("got gene inputs of size: {}".format(len(list_input_genes)))

    # translate the genes into what the system can handle
    list_input_translated = sql_utils.db_get_gene_curies_from_list(conn=sql_conn_query, list_input=list_input_genes)

    # return
    return list_input_translated


def process_genes(list_input_genes, p_value_cutoff, log=False):
    '''
    processes the input genes
    '''
    # initialize 
    sql_conn_query = sql_utils.db_sqlite_get_connection(db_path=db_file)

    # preprocess
    # translate the genes into what the system can handle
    list_input_translated = sql_utils.db_get_gene_names_from_list(conn=sql_conn_query, list_input=list_input_genes)
    logger.info("got translated gene inputs of size: {}".format(len(list_input_translated)))

    # do the calculations
    list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_gene_novelty, logs_process = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, 
                                                                                                               p_value=p_value_cutoff,
                                                                                                               list_gene=list_input_translated, 
                                                                                                               list_system_genes=list_system_genes, 
                                                                                                               map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
                                                                                                               mean_shifts=mean_shifts, scale_factors=scale_factors,
                                                                                                               log=True)
    
    # log
    for row in logs_process:
        logger.info(row)

    # return
    return map_gene_novelty, list_input_translated


def process_numeric_value(json_request, name, cutoff_default, is_float=True, log=False):
    '''
    will extract the p_value from the request; will return the default if none
    '''
    # initialize
    numeric_value = None

    # extract the value
    try:
        # Retrieve the float value from the request
        numeric_input = json_request.get(name)

        # Attempt to convert the input to a float
        if numeric_input is None:
            raise ValueError("No '{}' field provided in the request".format(name))

        if is_float:
            numeric_value = float(numeric_input)
        else:
            numeric_value = int(numeric_input)


    except ValueError as e:
        logger.error("got error for {}: {}".format(name, str(e)))
        numeric_value = cutoff_default

    except Exception as e:
        logger.error("got error for {}: {}".format(name, str(e)))
        numeric_value = cutoff_default

    # return
    return numeric_value


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)





    # # split the genes into list
    # if phenotypes:
    #     # build the phenotype list
    #     list_temp = phenotypes.split(",")
    #     list_select = []

    #     for value in list_temp:
    #         gene = value.strip()
    #         print("got phenotype: -{}-".format(gene))
    #         list_select.append(gene)

    #     # get the db connection
    #     conn = phenotype_utils.get_connection()

    #     # get the result diseases
    #     # map_disease = phenotype_utils.get_disease_score_for_phenotype_list(conn=conn, list_curies=list_select, log=False)
    #     list_disease = phenotype_utils.get_disease_score_sorted_list_for_phenotype_list(conn=conn, list_curies=list_select, log=False)
    #     print("got disease list size of: {}".format(len(list_disease)))

    #     # add to map
    #     # map_result['results'] = map_disease
    #     map_result['results'] = list_disease
