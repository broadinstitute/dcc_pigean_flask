
# imports
import pytest 
import json 
import networkx as nx

import dcc.graph_utils as gutils
import dcc.dcc_utils as dutils
import dcc.file_utils as futils
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 

# constants
path_files = "python-flask-server/conf/startup_files/"

# list_gene_set_file = ["gene_set_list_msigdb_c1.txt", "gene_set_list_msigdb_c2.txt", "gene_set_list_msigdb_c3.txt", "gene_set_list_msigdb_c4.txt", "gene_set_list_msigdb_c5.txt", 
#                       "gene_set_list_msigdb_c6.txt", "gene_set_list_msigdb_c7.txt", "gene_set_list_msigdb_c8.txt"]

list_gene_set_file = ["gene_set_list_msigdb_c1.txt", "gene_set_list_msigdb_c2.txt", "gene_set_list_msigdb_c3.txt", "gene_set_list_msigdb_c4.txt", "gene_set_list_msigdb_c5.txt", 
                      "gene_set_list_msigdb_c6.txt", "gene_set_list_msigdb_c7.txt", "gene_set_list_msigdb_c8.txt", "gene_set_list_msigdb_h.txt", "gene_set_list_msigdb_h2.txt",
                      "gene_set_list_mouse_2024.txt", "gene_set_list_ocr_human.txt"]
gene_file = "genes.txt"

# logger
logger = dutils.get_logger(name=__name__)


def get_factor_data():
    '''
    common method to get data
    '''
    print("\n=============== IN get_factor_data()")
    # initialize
    map_gene_index = None 
    list_input_genes = [
        "TCF7L2", "PPARG", "KCNJ11", "KCNQ1", "FTO", "SLC30A8", "HHEX", "CDKAL1",
        "IGF2BP2", "CDKN2A/B", "NOTCH2", "THADA", "ADAMTS9", "JAZF1", "CDC123/CAMK1D",
        "TP53INP1", "UBE2E2", "MAEA", "PRC1", "GCK", "GLIS3", "GCKR", "HNF1A",
        "HNF4A", "HNF1B", "MTNR1B", "ZNF365", "KLF14", "BCL11A", "GRB14", "HMGA2",
        "RREB1", "PPP1R3B", "PTPN1", "INS", "PAX4", "TCF2", "AP3S2", "APOB",
        "TSPAN8", "ADIPOQ", "ENPP1", "IRS1", "LMNA", "MAPK14", "NEUROD1", "SORCS1",
        "SUMO4", "TCF1", "WFS1", "ZNF259", "RFX6", "FADS1", "FADS2", "SREBF1",
        "PKHD1", "TCF4", "TRPM6", "INSR", "FOXO1", "MAP3K1", "MC4R", "NRXN3",
        "PDE4B", "PLCG1", "PPP1R3A", "PSEN1", "PSEN2", "PTEN", "RBP4", "REL",
        "RPL13", "RPS6", "SCD", "SLC2A2", "SLC9A9", "SLCO1B1", "SMAD3", "SOCS3",
        "SOX5", "SPHK1", "STXBP1", "TF", "TFAP2B", "TFAP2C", "TMEM18", "TNFAIP3",
        "TNNI3", "TOMM20", "TP53", "TRIB3", "TSLP", "UCP2", "UCP3", "UBE2L3",
        "UBQLN1", "UQCRC1", "VEGFA", "VLDLR", "WNT10B"
    ]

    # get the gene index
    map_gene_index, list_system_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # get the mean factors
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

    # get the factors
    list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_novelty, list_gene_set_p_values, logs_process = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, p_value=0.3,
                                                                                                               list_gene=list_input_genes, 
                                                                                                               list_system_genes=list_system_genes, 
                                                                                                               map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
                                                                                                               mean_shifts=mean_shifts, scale_factors=scale_factors,
                                                                                                               log=True)

    return list_factor, list_factor_genes, list_factor_gene_sets


# test methods
def test_build_factor_graph_for_llm():
    '''
    test generating new colors
    '''
    # get the factor data
    list_factor, list_factor_genes, list_factor_gene_sets = get_factor_data()

    # initialize
    graph = None

    # get the data
    graph = gutils.build_factor_graph_for_llm(list_factor=list_factor, list_factor_genes=list_factor_genes, 
                                              list_factor_gene_sets=list_factor_gene_sets,
                                              max_num_genes=3, max_num_gene_sets=2)

    # log
    logger.info("got network graph: {}".format(json.dumps(nx.node_link_data(graph), indent=2)))
    # print("got network graph: {}".format(json.dumps(nx.node_link_data(graph), indent=2)))

    # test
    assert graph is not None



