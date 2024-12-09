
# imports
import pytest 
import json 
import logging
import time
import json
import numpy as np

import dcc.file_utils as futils
import dcc.matrix_utils as mutils 
import dcc.compute_utils as cutils 
from dcc.spike_n_slab import SnS


# constants
# path_files = "python-flask-server/test/dcc/test_medium/"
# list_gene_set_file = ["gene_set_list_msigdb_c2.txt"]
path_files = "python-flask-server/conf/startup_files/"

# list_gene_set_file = ["gene_set_list_msigdb_c1.txt", "gene_set_list_msigdb_c2.txt", "gene_set_list_msigdb_c3.txt", "gene_set_list_msigdb_c4.txt", "gene_set_list_msigdb_c5.txt", 
#                       "gene_set_list_msigdb_c6.txt", "gene_set_list_msigdb_c7.txt", "gene_set_list_msigdb_c8.txt"]

# list_gene_set_file = ["gene_set_list_msigdb_c1.txt", "gene_set_list_msigdb_c2.txt", "gene_set_list_msigdb_c3.txt", "gene_set_list_msigdb_c4.txt", "gene_set_list_msigdb_c5.txt", 
#                       "gene_set_list_msigdb_c6.txt", "gene_set_list_msigdb_c7.txt", "gene_set_list_msigdb_c8.txt", "gene_set_list_msigdb_h.txt", "gene_set_list_msigdb_h2.txt",
#                       "gene_set_list_mouse_2024.txt", "gene_set_list_ocr_human.txt"]
list_gene_set_file = ["gene_set_list_msigdb_c1.txt", "gene_set_list_msigdb_c2.txt", "gene_set_list_msigdb_c3.txt"]
gene_file = "genes.txt"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# # test methods
# def test_gene_scores():
#     '''
#     test computing the factors
#     '''
#     print("\n=============== IN test_gene_scores()")
#     # initialize
#     map_gene_index = None 
#     list_input_genes = [
#         "TCF7L2", "PPARG", "KCNJ11", "KCNQ1", "FTO", "SLC30A8", "HHEX", "CDKAL1",
#         "IGF2BP2", "CDKN2A/B", "NOTCH2", "THADA", "ADAMTS9", "JAZF1", "CDC123/CAMK1D",
#         "TP53INP1", "UBE2E2", "MAEA", "PRC1", "GCK", "GLIS3", "GCKR", "HNF1A",
#         "HNF4A", "HNF1B", "MTNR1B", "ZNF365", "KLF14", "BCL11A", "GRB14", "HMGA2",
#         "RREB1", "PPP1R3B", "PTPN1", "INS", "PAX4", "TCF2", "AP3S2", "APOB",
#         "TSPAN8", "ADIPOQ", "ENPP1", "IRS1", "LMNA", "MAPK14", "NEUROD1", "SORCS1",
#         "SUMO4", "TCF1", "WFS1", "ZNF259", "RFX6", "FADS1", "FADS2", "SREBF1",
#         "PKHD1", "TCF4", "TRPM6", "INSR", "FOXO1", "MAP3K1", "MC4R", "NRXN3",
#         "PDE4B", "PLCG1", "PPP1R3A", "PSEN1", "PSEN2", "PTEN", "RBP4", "REL",
#         "RPL13", "RPS6", "SCD", "SLC2A2", "SLC9A9", "SLCO1B1", "SMAD3", "SOCS3",
#         "SOX5", "SPHK1", "STXBP1", "TF", "TFAP2B", "TFAP2C", "TMEM18", "TNFAIP3",
#         "TNNI3", "TOMM20", "TP53", "TRIB3", "TSLP", "UCP2", "UCP3", "UBE2L3",
#         "UBQLN1", "UQCRC1", "VEGFA", "VLDLR", "WNT10B"
#     ]

#     # get the gene index
#     map_gene_index, list_system_genes = futils.load_gene_file_into_map(file_path=path_files + gene_file)

#     # get the gene set/gene matrix data
#     matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

#     # get the gene vector
#     vector_gene, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_input_genes, map_gene_index=map_gene_index)

#     # get the mean factors
#     (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

#     # get the gene scores
#     # list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_novelty, list_gene_set_p_values, logs_process = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, p_value=0.3,
#     #                                                                                                            list_gene=list_input_genes, 
#     #                                                                                                            list_system_genes=list_system_genes, 
#     #                                                                                                            map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
#     #                                                                                                            mean_shifts=mean_shifts, scale_factors=scale_factors,
#     #                                                                                                            log=True)

#     # convert data for Alex's code
#     # make sure gene set matrix is full matrix, not sparse
#     matrix_dense_gene_sets = matrix_gene_sets.toarray()

#     # input the gene set/gene one hot matrix and gene vector
#     mod_log_sns = SnS()

#     # get log coeff
#     start = time.time()
#     logger.info("calculating beta tildes for gene scores")
#     log_coeff = mod_log_sns.compute_logistic_beta_tildes(X=matrix_dense_gene_sets, Y=vector_gene)
#     # log_coeff_beta_tildes, log_coeff_ses, z_scores, p_values, se_inflation_factors
#     # log_coeff_beta_tildes, log_coeff_ses, _, _, _ = mod_log_sns.compute_logistic_beta_tildes(X=matrix_dense_gene_sets, Y=vector_gene)
#     end = time.time()
#     str_message = "log coeff calculation time elapsed {}s".format(end-start)
#     logger.info(str_message)
#     # logger.info("got log coeff: {}".format(log_coeff_beta_tildes))
#     logger.info("got log coeff: {}".format(log_coeff))


#     # get log coeff using member vars
#     # start = time.time()
#     # logger.info("calculating beta tildes for gene scores")
#     # log_coeff_beta_tildes, log_coeff_ses, _, _, _, _, _ = mod_log_sns.compute_logistic_beta_tildes(X=matrix_dense_gene_sets, Y=vector_gene)
#     # end = time.time()
#     # str_message = "log coeff calculation time elapsed {}s".format(end-start)
#     # logger.info(str_message)
#     # logger.info("got log coeff: {}".format(log_coeff_beta_tildes))

#     # get gene scores
#     # gene_betas, _ = mod_log_sns.calculate_non_inf_betas(
#     #                 log_coeff[0][:, features_to_keep],
#     #                 log_coeff[1][:, features_to_keep],
#     #                 X_orig=logit_data[:, features_to_keep],
#     #                 sigma2=np.ones((1, 1)) * sigma_reg,
#     #                 p=np.ones((1, 1)) * 0.001, assume_independent=assume_independent)
#     gene_betas, _ = mod_log_sns.calculate_non_inf_betas(
#                     log_coeff[0][:, list_input_gene_indices],
#                     log_coeff[1][:, list_input_gene_indices],
#                     X_orig=matrix_dense_gene_sets[:, list_input_gene_indices],
#                     sigma2=np.ones((1, 1)),
#                     p=np.ones((1, 1)) * 0.001)
#     start = time.time()
#     logger.info("calculating gene scores for gene scores")
#     log_coeff_beta_tildes, log_coeff_ses, _, _, _, _, _ = mod_log_sns.compute_logistic_beta_tildes(X=matrix_dense_gene_sets, Y=vector_gene)
#     end = time.time()
#     str_message = "gene scores calculation time elapsed {}s".format(end-start)
#     logger.info(str_message)
#     logger.info("got gene scores: {}".format(gene_betas))

#     # test
#     assert log_coeff is not None



def test_gene_scores_compute_lib():
    '''
    test computing the factors
    '''
    print("\n=============== IN test_gene_scores_compute_lib()")
    # initialize
    map_gene_index = None 
    map_gene_scores = None
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

    # get the gene set/gene matrix data
    matrix_gene_sets, map_gene_set_index = mutils.load_geneset_matrix(map_gene_index=map_gene_index, list_gene_set_files=list_gene_set_file, path_gene_set_files=path_files, log=False)

    # get the gene vector
    vector_gene, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_input_genes, map_gene_index=map_gene_index)

    # get the mean factors
    logger.info("UNIT TEST - calculating mean_shifts, scale_factors for gene scores")
    (mean_shifts, scale_factors) = cutils._calc_X_shift_scale(X=matrix_gene_sets)

    # get the gene scores
    # list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_novelty, list_gene_set_p_values, logs_process = cutils.calculate_factors(matrix_gene_sets_gene_original=matrix_gene_sets, p_value=0.3,
    #                                                                                                            list_gene=list_input_genes, 
    #                                                                                                            list_system_genes=list_system_genes, 
    #                                                                                                            map_gene_index=map_gene_index, map_gene_set_index=map_gene_set_index,
    #                                                                                                            mean_shifts=mean_shifts, scale_factors=scale_factors,
    #                                                                                                            log=True)

    start = time.time()

    # OLD
    # map_gene_scores = cutils.calculate_gene_scores_map(matrix_gene_sets=matrix_gene_sets, list_input_genes=list_input_genes, map_gene_index=map_gene_index, list_system_genes=list_system_genes)

    # get the p_values, beta_tildes and ses
    logger.info("UNIT TEST - calculating p_values for gene scores")
    vector_gene_set_pvalues, vector_beta_tildes, vector_ses = cutils.compute_beta_tildes(X=matrix_gene_sets, Y=vector_gene, scale_factors=scale_factors, mean_shifts=mean_shifts)

    logger.info("UNIT TEST - getting gene score map for gene scores")
    map_gene_set_scores = cutils.calculate_gene_scores_map(matrix_gene_sets=matrix_gene_sets, vector_gene=vector_gene, list_input_genes=list_input_genes, map_gene_index=map_gene_index, list_system_genes=list_system_genes,
                                                       input_p_values=vector_gene_set_pvalues, input_beta_tildes=vector_beta_tildes, input_ses=vector_beta_tildes, 
                                                       input_scale_factors=scale_factors, log=True)
    
    end = time.time()
    str_message = "gene scores calculation time elapsed {}s".format(end-start)
    logger.info(str_message)
    logger.info("got gene scores: {}".format(map_gene_set_scores))

    # log
    logger.info("got gene scores: {}".format(json.dumps(map_gene_set_scores, indent=2)))

    # test
    assert map_gene_scores is not None
    assert len(map_gene_scores) > 0
