# imports
from flask import Flask, render_template, request, flash

from startup_utils import load_conf


# constants
DEBUG=True

# variables
app = Flask(__name__)
app.secret_key = "test_app_gpt"

map_conf = load_conf()

@app.route("/query", methods=["POST"])
def post_genes():
    # initialize 
    map_result = {}
    phenotypes = None
    list_genes = []
    genes = None 

    # get the input
    data = request.get_json()
    if data:
        list_genes = str(data.get('genes'))

    print("got request: {} with gene inputs: {}".format(request.method, list_genes))

    # add the genes to the result
    map_result['input_genes'] = list_genes
    if DEBUG:
        map_result['conf'] = map_conf

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
