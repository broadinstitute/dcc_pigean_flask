# imports
from flask import Flask, render_template, request, flash
from sql_utils import get_connection, get_list_abstracts, get_map_gene_abstracts
import ml_utils

app = Flask(__name__)
app.secret_key = "test_app_gpt"

@app.route("/genes")
def index():
    return render_template("index.html")


@app.route("/submit_genes", methods=["POST","GET"])
def submit_genes():
    # initialize 
    list_genes = []
    list_genes_missing = []
    list_temp = []
    list_abstracts = []
    abstract_gpt = "no abstract"
    prompt_gpt = "no prompt"
    list_gene_llm = []
    list_abstract_llm = []
    biology_abstract = ""
    pathway_abstract = ""
    input_genes = ""
    map_abstracts = {}

    # get the input
    if request.method == 'POST':
        input_genes = str(request.form["input_gene_set"])
  
    else:
        if request.args.get('input_gene_set'):
            input_genes = str(request.args.get('input_gene_set'))

    print("got request: {} with inputs: {}".format(request.method, input_genes))


    # split the genes into list
    if input_genes:
        list_temp = input_genes.split(",")
        list_select = []

        for value in list_temp:
            gene = value.strip()
            print("got gene: -{}-".format(gene))
            list_select.append(gene)

        # get the data
        conn = get_connection()
        # list_abstracts = get_list_abstracts(conn=conn, list_genes=list_select, log=True)
        map_gene_abstracts = get_map_gene_abstracts(conn=conn, list_genes=list_select, log=True)

        if map_gene_abstracts and len(map_gene_abstracts) > 0:
            # build the prompt inputs
            str_gene = ",".join(map_gene_abstracts.keys())
            # str_abstract = "\n".join(list_abstract_llm)

            print("found summaries for genes: {}".format(str_gene))
            # print("\ngot abstracts: {}".format(str_abstract))

            # call the LLM
            # biology_abstract = ml_utils.call_llm(prompt_template=ml_utils.PROMPT_BIOLOGY, str_gene=str_gene, str_abstract=str_abstract, log=True)
            biology_abstract = ml_utils.call_gene_abstract_llm_recurisve(prompt_template=ml_utils.PROMPT_BIOLOGY, map_gene_abstracts=map_gene_abstracts, max_tokens=2500, log=True)
            print("\n\ngot biology LLM result: \n{}".format(biology_abstract))

            list_gene_llm = list(map_gene_abstracts.keys())
            # # call the LLM
            # pathway_abstract = ml_utils.call_llm(prompt_template=ml_utils.PROMPT_PATHWAYS, str_gene=str_gene, str_abstract=str_abstract, log=True)
            # print("\n\ngot biology LLM result: \n{}".format(biology_abstract))

            # call the defaultt chatGPT answer
            biology_no_abstracts = ml_utils.call_llm_no_abstracts(str_gene=str_gene, log=True)

    else:
        print("no input genes")

    # add data for return 
    flash(list_select, 'list_genes_input')
    flash(list_gene_llm, 'list_genes_used')
    flash(biology_abstract, 'abstract_biology')
    flash(biology_no_abstracts, 'no_abstract_biology')
    # flash(pathway_abstract, 'abstract_pathway')
    # flash(list_genes_missing, 'list_missing')
    # flash(prompt_gpt, 'prompt')
    # flash(abstract_gpt, 'abstract')

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
