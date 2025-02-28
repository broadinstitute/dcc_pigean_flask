import numpy as np

import dcc.matrix_utils as mutils  
import dcc.sql_utils as sql_utils
import dcc.startup_utils as sutils
import dcc.compute_utils as cutils


def run_phewas():
    map_conf = sutils.load_conf()
    gene_phewas_bfs_in = map_conf.get('root_dir') +  "all_trait_gcat.trait_gene_combined.gt1.txt"
    db_file = map_conf.get('root_dir') +  map_conf.get('db_file')
    sql_connection = sql_utils.db_sqlite_get_connection(db_path=db_file)
    map_gene_index, list_system_genes, map_gene_ontology = sql_utils.db_load_gene_table_into_map(conn=sql_connection)
    phenos, gene_pheno_Y, gene_pheno_combined_prior_Ys = mutils.read_gene_phewas_bfs(list_system_genes, map_gene_index, gene_phewas_bfs_in)
    gene_list = ['PPARG','LMNA','CIDEC','LIPE','PLIN1','AKT2']
    p_values, beta_tildes, ses = cutils.calculate_phewas(gene_list, list_system_genes, map_gene_index, phenos, gene_pheno_Y, gene_pheno_combined_prior_Ys)

    with open('phewas_results.txt', 'w') as f:
        print('i', 'phenos', 'p_values', 'beta_tildes', 'ses', sep='\t', file=f)
        for i in range(len(phenos)):
           print(i, phenos[i], p_values[i], beta_tildes[i], ses[i], sep='\t', file=f)

def main():
    run_phewas()

if __name__ == '__main__':
    main()
