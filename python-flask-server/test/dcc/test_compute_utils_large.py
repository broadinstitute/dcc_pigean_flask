
# imports
import pytest 
import json 
import logging
import numpy as np

import dcc.file_utils as futils
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 


# constants
path_files = "python-flask-server/test/dcc/test_medium/"
list_gene_set_file = ["gene_set_list_msigdb_c2.txt"]
gene_file = "genes.txt"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# test methods
def test_calculate_factors():
    '''
    test computing the factors
    '''
    print("\n=============== IN test_calculate_factors()")
    # initialize
    map_gene_index = None 
    list_genes = [
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
    map_gene_index = futils.load_gene_file_into_map(file_path=path_files + gene_file)

    # get the data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # get the mean factors
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

    # get the factors
    gene_factors, gene_set_factors, map_filtered_index, map_filtered_gene_set_index = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, list_gene=list_genes, 
                                                                                                               map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
                                                                                                               mean_shifts=mean_shifts, scale_factors=scale_factors,
                                                                                                               log=True)

    # test
    assert gene_factors is not None
    assert gene_set_factors is not None


