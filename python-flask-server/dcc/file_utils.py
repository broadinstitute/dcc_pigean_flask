
# imports
import json

# methods
def read_tab_delimited_file(file_path, log=False):
    '''
    will read the pathway file and build a map of gene lists
    '''
    # initialize
    result = {}

    # read in the data
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            stripped_list = [item.strip() for item in fields]

            key = fields[0]
            values = stripped_list[1:]
            if key in result:
                result[key].extend(values)
            else:
                result[key] = values

    # return
    return result


def load_gene_file_into_map(file_path, log=False):
    '''
    will read the gene file and return a map of gene to position in array
    '''
    # initialize
    map_result = {}
    list_temp = []

    # read in the data
    with open(file_path, 'r') as file:
        for line in file:
            gene = line.strip()
            list_temp.append(gene)
            # result[gene] = count
            # count = count + 1

    # create the map from the list; make sure no duplicates
    list_unique = list(set(list_temp))
    list_unique.sort()
    map_result = {value: index for index, value in enumerate(list_unique)}

    # return
    return map_result




if __name__ == "__main__":

    file_path = '/home/javaprog/Code/TranslatorWorkspace/PigeanFlask/python-flask-server/conf/gene_lists/gene_set_list_msigdb_h.txt'
    data = read_tab_delimited_file(file_path)
    print(json.dumps(data, indent=2))