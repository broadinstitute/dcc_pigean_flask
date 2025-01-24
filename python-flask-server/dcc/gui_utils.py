
# LICENSE
# Copyright 2024 Flannick Lab

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

# imports
import dcc.matrix_utils as mutils 
import dcc.dcc_utils as dutils 
import dcc.graph_utils as gutils
import dcc.data_utils as dautils 


# constants
logger = dutils.get_logger(__name__)


# methods
def gui_build_results_map(list_factor, list_factor_genes, list_factor_gene_sets, map_gene_ontology, list_input_gene_names, map_gene_index, matrix_gene_sets, map_gene_factor_data, log=False):
    '''
    builds the map results
    '''
    # initialize
    map_result = {}
    num_results = dutils.NUMBER_RETURNED_PER_FACTOR

    # add the data
    list_result = []

    # loop through the factors
    for index, row in enumerate(list_factor):
        list_result.append({'top_set': row, 'gene_sets': list_factor_gene_sets[index][:num_results], 'genes': list_factor_genes[index][:num_results]})

    # add the factors to the map
    map_result['data'] = list_result

    # log
    logger.info("added factor data to REST of size: {}".format(len(list_result)))

    # add the input genes
    map_result_genes = {}
    for gene_name in list_input_gene_names:
        # get the ontology_id
        ontology_id = map_gene_ontology.get(gene_name)
        if ontology_id:
            map_result_genes[ontology_id] = {'name': gene_name, 'ontology_id': ontology_id}

            # annotate the gene data
            index_gene = map_gene_index.get(gene_name)
            if index_gene:
                # get the number of pathways the gene is in
                count_pathways = mutils.sum_of_gene_row(sparse_matrix=matrix_gene_sets, gene_index=index_gene).item()
                map_result_genes.get(ontology_id)['count_pathways'] = count_pathways

                # add the novelty score, hightest factor name and highest factor score to the gene
                gene_factor_data_map = map_gene_factor_data.get(gene_name)
                if not gene_factor_data_map:
                    map_result_genes.get(ontology_id)['novelty_score'] = 0.0
                    map_result_genes.get(ontology_id)[dutils.KEY_INTERNAL_HIGHEST_FACTOR_NAME] = None
                    map_result_genes.get(ontology_id)[dutils.KEY_INTERNAL_HIGHEST_FACTOR_SCORE] = None
                else:
                    map_result_genes.get(ontology_id)['novelty_score'] = gene_factor_data_map.get(dutils.KEY_INTERNAL_LOWEST_FACTOR_SCORE)
                    map_result_genes.get(ontology_id)[dutils.KEY_INTERNAL_HIGHEST_FACTOR_NAME] = gene_factor_data_map.get(dutils.KEY_INTERNAL_HIGHEST_FACTOR_NAME)
                    map_result_genes.get(ontology_id)[dutils.KEY_INTERNAL_HIGHEST_FACTOR_SCORE] = gene_factor_data_map.get(dutils.KEY_INTERNAL_HIGHEST_FACTOR_SCORE)


                if log:
                    logger.info("for gene: {}, got index: {}, pathway count: {}, novelty score: {}".format(gene_name, index_gene, count_pathways, map_result_genes.get(ontology_id).get('novelty_score')))


    logger.info("got novelty gene map of size: {}".format(len(map_gene_factor_data)))
    logger.info("returning result gene map of size: {}".format(len(map_result_genes)))

    # add the input genes to the map
    map_result['input_genes'] = list_input_gene_names
    map_result['input_genes_map'] = map_result_genes

    # return
    return map_result


def gui_build_novelty_results_map(map_gene_ontology, list_input_gene_names, map_gene_index, matrix_gene_sets, map_gene_factor_data, log=False):
    '''
    builds the map results
    '''
    # initialize
    map_result = {}

    # add the input genes
    map_result_genes = {}
    for gene_name in list_input_gene_names:
        # get the ontology_id
        ontology_id = map_gene_ontology.get(gene_name)
        if ontology_id:
            map_result_genes[ontology_id] = {'name': gene_name, 'ontology_id': ontology_id}

            # annotate the gene data
            index_gene = map_gene_index.get(gene_name)
            if index_gene:
                # get the number of pathways the gene is in
                count_pathways = mutils.sum_of_gene_row(sparse_matrix=matrix_gene_sets, gene_index=index_gene).item()
                map_result_genes.get(ontology_id)['count_pathways'] = count_pathways

                # add the novelty score to the gene
                novelty_score = map_gene_factor_data.get(gene_name).get(dutils.KEY_INTERNAL_LOWEST_FACTOR_SCORE)
                if not novelty_score:
                    novelty_score = 0.0
                map_result_genes.get(ontology_id)['novelty_score'] = novelty_score

                if log:
                    logger.info("for gene: {}, got index: {}, pathway count: {}, novelty score: {}".format(gene_name, index_gene, count_pathways, novelty_score))


    logger.info("got novelty gene map of size: {}".format(len(map_gene_factor_data)))
    logger.info("returning result gene map of size: {}".format(len(map_result_genes)))

    # add the input genes to the map
    map_result['gene_results'] = map_result_genes

    # return
    return map_result


def gui_build_pigean_app_results_map(list_input_genes, list_factor, list_factor_genes, list_factor_gene_sets, list_gene_set_p_values, max_num_per_factor=dutils.NUMBER_RETURNED_PER_FACTOR, log=False):
    '''
    root method to build the pigean app results 
    '''
    # initialize
    map_result = {}

    # get the index values of the clean factors
    list_verified_index = get_list_verified_results_for_giu(list_factor=list_factor, list_factor_genes=list_factor_genes, list_factor_gene_sets=list_factor_gene_sets)
    list_factor_gui = [list_factor[index] for index in list_verified_index]
    list_factor_genes_gui = [list_factor_genes[index] for index in list_verified_index]
    list_factor_gene_sets_gui = [list_factor_gene_sets[index] for index in list_verified_index]

    # build the subsets of the data
    pigean_factor_map = build_pigean_factor_results_map(list_factor=list_factor_gui, list_factor_genes=list_factor_genes_gui, list_factor_gene_sets=list_factor_gene_sets_gui, 
            max_num_per_factor=max_num_per_factor, log=log)
    map_result[dutils.KEY_APP_FACTOR_PIGEAN] = pigean_factor_map

    # add the gene factors
    map_result[dutils.KEY_APP_FACTOR_GENE] = build_pigean_gene_factor_results_map(list_factor=list_factor_gui, list_factor_genes=list_factor_genes_gui, list_factor_gene_sets=list_factor_gene_sets_gui)

    # add the gene factors
    map_result[dutils.KEY_APP_FACTOR_GENE_SET] = build_pigean_gene_set_factor_results_map(list_factor=list_factor_gui, list_factor_gene_sets=list_factor_gene_sets_gui)

    # add the input gene list
    map_result[dutils.KEY_APP_INPUT_GENES] = list_input_genes

    # add the gene set p_value list list
    map_result[dutils.KEY_APP_GENE_SETS] = list_gene_set_p_values

    # return
    return map_result


def gui_build_gene_scores_app_results_map(map_gene_scores, map_gene_set_scores, log=False):
    '''
    root method to build the pigean app results 
    '''
    # initialize
    map_result = {}

    # add the gene scores
    map_result[dutils.KEY_APP_GENE_SCORES] = map_gene_scores

    # add the gene set scores
    map_result[dutils.KEY_APP_GENE_SET_SCORES] = map_gene_set_scores

    # return
    return map_result


def get_list_verified_results_for_giu(list_factor, list_factor_genes, list_factor_gene_sets, log=False):
    '''
    returns the indexes of the verified factor results 
    '''
    list_result = []

    print("/n/n/nlist factors: {}".format(list_factor))

    for index, row in enumerate(list_factor):
        if row and row.get(dutils.KEY_APP_GENE_SET):
            list_result.append(index)

    # return
    return list_result


def build_pigean_gene_factor_results_map(list_factor, list_factor_genes, list_factor_gene_sets, log=False):
    '''
    builds the map results for the pigean app gene factor subset
    '''
    # initialize
    map_result = {}

    # loop through the factors and add a list per gene
    # loop through the factors
    for index, row in enumerate(list_factor):
        # BUG - only add factors if gene set and genes are not null
        # # See GitHub issue #2 for more details.
        if row['gene_set']:
            name = "Factor{}".format(index)
            label = list_factor[index][dutils.KEY_APP_GENE_SET]
            list_temp = []

            # loop through the gene factors for this factor and add to list
            for map_gene in list_factor_genes[index]:
                list_temp.append({'label_factor': name, 'gene': map_gene.get('gene'), 'factor_value': map_gene.get('score'), 'label': label})

            map_result[name] = list_temp

    # return 
    return map_result


def build_pigean_gene_set_factor_results_map(list_factor, list_factor_gene_sets, log=False):
    '''
    builds the map results for the pigean app gene factor subset
    '''
    # initialize
    map_result = {}

    # loop through the factors and add a list per gene
    # loop through the factors
    for index, row in enumerate(list_factor):
        # BUG - only add factors if gene set and genes are not null
        # # See GitHub issue #2 for more details.
        if row['gene_set']:
            name = "Factor{}".format(index)
            label = list_factor[index][dutils.KEY_APP_GENE_SET]
            list_temp = []

            # loop through the gene factors for this factor and add to list
            for map_gene_set in list_factor_gene_sets[index]:
                list_temp.append({'label_factor': name, 'gene_set': map_gene_set.get('gene_set'), 'factor_value': map_gene_set.get('score'), 'label': label})

            map_result[name] = list_temp

    # return 
    return map_result


def build_pigean_factor_results_map(list_factor, list_factor_genes, list_factor_gene_sets, max_num_per_factor, log=False):
    '''
    builds the map results for the pigean app factor subset
    '''
    # initialize
    map_result = {}
    list_result = []

    # loop through the factors
    for index, row in enumerate(list_factor):
        # BUG - only add factors if gene set and genes are not null
        # # See GitHub issue #2 for more details.
        if row['gene_set']:
            name = "Factor{}".format(index)
            map_temp = {'cluster': name, 'factor': name, 'label': row['gene_set']}

            # create top genes and gene sets as ; delimited string
            map_temp['top_genes'] = ';'.join([item['gene'] for item in list_factor_genes[index][:max_num_per_factor] if item['gene'] is not None])
            map_temp['top_gene_sets'] = ';'.join([item['gene_set'] for item in list_factor_gene_sets[index][:max_num_per_factor] if item['gene_set'] is not None])

            print("list genes: {}".format(list_factor_genes))
            if len(list_factor_genes[index]) > 0:
                map_temp['gene_score'] = max([item['score'] if item['score'] else 0 for item in list_factor_genes[index]])
            else:
                map_temp['gene_score'] = 0

            if len(list_factor_gene_sets[index]) > 0:
                map_temp['gene_set_score'] = max([item['score'] if item['score'] else 0 for item in list_factor_gene_sets[index]])
            else:
                map_temp['gene_set_score'] = 0

            # add to list
            list_result.append(map_temp)

    # add the factors to the map
    map_result['data'] = list_result

    # return
    return map_result


def build_graph_node_edge_map(list_factor_input, list_factor_genes_input, list_factor_gene_sets_input, log=False):
    '''
    will build the map for the VIS graph display library format
    '''
    map_result = {}
    graph = None
    list_nodes = []
    list_edges = []
    
    # # get the manipulated list factor list
    # list_factor = dautils.extract_factor_data_list(list_factor_input=list_factor_input, list_factor_genes_input=list_factor_genes_input, list_factor_gene_sets_input=list_factor_gene_sets_input)

    # get the network x graph object
    graph = gutils.build_factor_graph(list_factor=list_factor_input, list_factor_genes=list_factor_genes_input, list_factor_gene_sets=list_factor_gene_sets_input)

    # get the data
    data = gutils.extract_data_from_graph(graph=graph)
    map_result['networkx'] = data

    # get the edges and nodes
    list_nodes, list_edges = gutils.extract_nodes_edges_from_graph(graph=graph)

    # log
    if log:
        logger.info("got graph nodes: {}".format(list_nodes))    
        logger.info("got graph edges: {}".format(list_edges))    

    # add to map
    map_result['data'] = [{'nodes': list_nodes, 'edges': list_edges}]

    # return
    return map_result


# main
if __name__ == "__main__":
    pass