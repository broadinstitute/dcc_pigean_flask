# imports
from flask import Flask, render_template, request, flash

import startup_utils as sutils
import file_utils as futils
import matrix_utils as mutils 
import compute_utils as cutils 
import dcc_utils as dutils 


# constants
DEBUG=False

# variables
app = Flask(__name__)
app.secret_key = "test_app_gpt"

logger = dutils.get_logger(__name__)
p_value_cutoff = 0.3

# in memory compute variables
map_conf = sutils.load_conf()
map_gene_index, list_system_genes = futils.load_gene_file_into_map(file_path=map_conf.get('root_dir') + map_conf.get('gene_file'))
matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, 
                                                                  list_gene_set_files=map_conf.get('gene_set_files'), path_gene_set_files=map_conf.get('root_dir'), log=False)
(mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)



@app.route("/query", methods=["POST"])
def post_genes():
    # initialize 
    map_result = {}
    list_input_genes = []

    # get the input
    data = request.get_json()
    if data:
        list_input_genes = data.get('genes')

    print("got request: {} with gene inputs: {}".format(request.method, list_input_genes))

    # add the genes to the result
    map_result['input_genes'] = list_input_genes
    if DEBUG:
        map_result['conf'] = map_conf

    # compute
    list_factor, list_factor_genes, list_factor_gene_sets = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, p_value=0.3,
                                                                                                               list_gene=list_input_genes, 
                                                                                                               list_system_genes=list_system_genes, 
                                                                                                               map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
                                                                                                               mean_shifts=mean_shifts, scale_factors=scale_factors,
                                                                                                               log=True)

    # format the data
    map_factors = cutils.group_factor_results(list_factor=list_factor, list_factor_gene_sets=list_factor_gene_sets, list_factor_genes=list_factor_genes)
    map_result['data'] = map_factors


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

    # return
    return map_result



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)
