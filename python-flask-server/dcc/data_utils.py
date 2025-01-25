
# imports
import dcc.dcc_utils as dutils 


# constants
logger = dutils.get_logger(__name__)

# methods
def get_list_verified_results_for_giu(list_factor, list_factor_genes, list_factor_gene_sets, log=False):
    '''
    returns the indexes of the verified factor results 
    '''
    list_result = []

    # print("/n/n/nlist factors: {}".format(list_factor))

    for index, row in enumerate(list_factor):
        if row and row.get(dutils.KEY_APP_GENE_SET):
            list_result.append(index)

    # return
    return list_result


def extract_factor_data_list(list_factor_input, list_factor_genes_input, list_factor_gene_sets_input, max_num_per_factor=5, log=True):
    '''
    builds the list results for the pigean app factor subset
    '''
    # initialize
    list_result = []

    # get the index values of the clean factors
    list_verified_index = get_list_verified_results_for_giu(list_factor=list_factor_input, list_factor_genes=list_factor_genes_input, list_factor_gene_sets=list_factor_gene_sets_input)
    list_factor = [list_factor_input[index] for index in list_verified_index]
    list_factor_genes = [list_factor_genes_input[index] for index in list_verified_index]
    list_factor_gene_sets = [list_factor_gene_sets_input[index] for index in list_verified_index]

    # loop through the factors
    for index, row in enumerate(list_factor):
        # BUG - only add factors if gene set and genes are not null
        # # See GitHub issue #2 for more details.
        if row['gene_set']:
            name = "Factor{}".format(index)
            map_temp = {'cluster': name, 'factor': name, 'label': row['gene_set']}

            # log
            if log:
                logger.info("extracting factor for GUI data of name: {}".format(map_temp))

            # create top genes and gene sets as ; delimited string
            map_temp['top_genes'] = ';'.join([item['gene'] for item in list_factor_genes[index][:max_num_per_factor] if item['gene'] is not None])
            map_temp['top_gene_sets'] = ';'.join([item['gene_set'] for item in list_factor_gene_sets[index][:max_num_per_factor] if item['gene_set'] is not None])

            # print("list genes: {}".format(list_factor_genes))
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

    # return
    return list_result


def extract_pigean_gene_factor_results_map(list_factor, list_factor_genes, max_num_per_factor=10, log=False):
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

            # loop through the gene factors for this factor and add to list
            for map_gene in list_factor_genes[index][0:max_num_per_factor]:
                label_gene = map_gene.get('gene')
                if not map_result.get(label_gene):
                    map_result[label_gene] = []

                map_result[label_gene].append({'factor': name, 'factor_score': map_gene.get('score')})

    # return 
    return map_result


def extract_pigean_gene_set_factor_results_map(list_factor, list_factor_gene_sets, max_num_per_factor=10, log=False):
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

            # loop through the gene factors for this factor and add to list
            for map_gene_set in list_factor_gene_sets[index][0:max_num_per_factor]:
                label_gene_set = map_gene_set.get('gene_set')
                if not map_result.get(label_gene_set):
                    map_result[label_gene_set] = []

                map_result[label_gene_set].append({'factor': name, 'factor_score': map_gene_set.get('score')})

    # return 
    return map_result
