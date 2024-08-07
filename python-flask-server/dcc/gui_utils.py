

# imports
import dcc.matrix_utils as mutils 
import dcc.dcc_utils as dutils 



# constants
logger = dutils.get_logger(__name__)


# methods
def gui_build_results_map(list_factor, list_factor_genes, list_factor_gene_sets, map_gene_ontology, list_input_gene_names, map_gene_index, matrix_gene_sets, map_gene_novelty, log=False):
    '''
    builds the map results
    '''
    # initialize
    map_result = {}

    # add the data
    list_result = []

    # loop through the factors
    for index, row in enumerate(list_factor):
        list_result.append({'top_set': row, 'gene_sets': list_factor_gene_sets[index], 'genes': list_factor_genes[index]})

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

                # add the novelty score to the gene
                novelty_score = map_gene_novelty.get(gene_name)
                if not novelty_score:
                    novelty_score = 0.0
                map_result_genes.get(ontology_id)['novelty_score'] = novelty_score

                if log:
                    logger.info("for gene: {}, got index: {}, pathway count: {}, novelty score: {}".format(gene_name, index_gene, count_pathways, novelty_score))


    logger.info("got novelty gene map of size: {}".format(len(map_gene_novelty)))
    logger.info("returning result gene map of size: {}".format(len(map_result_genes)))

    # add the input genes to the map
    map_result['input_genes'] = list_input_gene_names
    map_result['input_genes_map'] = map_result_genes

    # return
    return map_result

def gui_build_novelty_results_map(map_gene_ontology, list_input_gene_names, map_gene_index, matrix_gene_sets, map_gene_novelty, log=False):
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
                novelty_score = map_gene_novelty.get(gene_name)
                if not novelty_score:
                    novelty_score = 0.0
                map_result_genes.get(ontology_id)['novelty_score'] = novelty_score

                if log:
                    logger.info("for gene: {}, got index: {}, pathway count: {}, novelty score: {}".format(gene_name, index_gene, count_pathways, novelty_score))


    logger.info("got novelty gene map of size: {}".format(len(map_gene_novelty)))
    logger.info("returning result gene map of size: {}".format(len(map_result_genes)))

    # add the input genes to the map
    map_result['gene_results'] = map_result_genes

    # return
    return map_result


# main
if __name__ == "__main__":
    pass