"""Test Clinical Evidence function."""

# imports
import pytest 
import logging
import sys
import os
import json

# add path of file
import sys
import os

# Determine the path to startup_utils.py
startup_utils_path = os.path.abspath("python-flask-server/gene_nmf_adapter.py")
startup_utils_dir = os.path.dirname(startup_utils_path)
sys.path.insert(0, startup_utils_dir)
print("added to pyton path the following directory: {}".format(startup_utils_dir))

# import file top test
import gene_nmf_adapter as adapter
import dcc.dcc_utils as dutils

# logger
logger = dutils.get_logger(name=__name__)
logger.info("added to pyton path the following directory: {}".format(startup_utils_dir))

# vars
list_gene_test  = [
    "UTP14A", "S100A6", "SCAND1", "RRP12", "CIAPIN1", "ADH5", "MTERF3", "SPR", "CHMP4A",
    "UFM1", "VAT1", "HACD3", "RFC5", "COTL1", "NPRL2", "TRIB3", "PCCB", "TLE1", "CD58",
    "BACE2", "KDM3A", "TARBP1", "RNH1", "CHAC1", "MBNL2", "VDAC1", "TES", "OXA1L", "NOP56",
    "HAT1", "CPNE3", "DNMT1", "ARHGAP1", "VPS28", "EIF2S2", "BAG3", "CDCA4", "NPDC1", "RPS6KA1",
    "FIS1", "SYPL1", "SARS", "CDC45", "CANT1", "HERPUD1", "SORBS3", "MRPS2", "TOR1A", "TNIP1",
    "SLC25A46", "MAL", "EPCAM", "HDAC6", "CAPN1", "TNRC6B", "PKD1", "RRS1", "HP", "ANO10",
    "CEP170B", "IDE", "DENND2D", "CAMK2B", "ZNF358", "RPP38", "MRPL19", "NUCB2", "GNAI1", "LSR",
    "ADGRE2", "PKMYT1", "CDK5R1", "ABL1", "PILRB", "AXIN1", "FBXL8", "MCF2L", "DBNDD1", "IGHMBP2",
    "WIPF2", "WFS1", "OGFOD2", "MAPK1IP1L", "COL11A1", "REG3A", "SERPINA1", "MYCBP2", "PIGK", "TCAP",
    "CRADD", "ELK1", "DNAJB2", "ZBTB16", "DAZAP1", "MAPKAPK2", "EDRF1", "CRIP1", "UCP3", "AGR2",
    "P4HA2"
]

# NOTE - t2d genes
# list_gene_test  = [
#         "TCF7L2",
#         "PPARG",
#         "KCNJ11",
#         "KCNQ1",
#         "FTO",
#         "SLC30A8",
#         "HHEX",
#         "CDKAL1",
#         "IGF2BP2",
#         "CDKN2A/B",
#         "NOTCH2",
#         "THADA",
#         "ADAMTS9",
#         "JAZF1",
#         "CDC123/CAMK1D",
#         "TP53INP1",
#         "UBE2E2",
#         "MAEA",
#         "PRC1",
#         "GCK",
#         "GLIS3",
#         "GCKR",
#         "HNF1A",
#         "HNF4A",
#         "HNF1B",
#         "MTNR1B",
#         "ZNF365",
#         "KLF14",
#         "BCL11A",
#         "GRB14",
#         "HMGA2",
#         "RREB1",
#         "PPP1R3B",
#         "PTPN1",
#         "INS",
#         "PAX4",
#         "TCF2",
#         "AP3S2",
#         "APOB",
#         "TSPAN8",
#         "ADIPOQ",
#         "ENPP1",
#         "IRS1",
#         "LMNA",
#         "MAPK14",
#         "NEUROD1",
#         "SORCS1",
#         "SUMO4",
#         "TCF1",
#         "WFS1",
#         "ZNF259",
#         "RFX6",
#         "FADS1",
#         "FADS2",
#         "SREBF1",
#         "PKHD1",
#         "TCF4",
#         "TRPM6",
#         "INSR",
#         "FOXO1",
#         "MAP3K1",
#         "MC4R",
#         "NRXN3",
#         "PDE4B",
#         "PLCG1",
#         "PPP1R3A",
#         "PSEN1",
#         "PSEN2",
#         "PTEN",
#         "RBP4",
#         "REL",
#         "RPL13",
#         "RPS6",
#         "SCD",
#         "SLC2A2",
#         "SLC9A9",
#         "SLCO1B1",
#         "SMAD3",
#         "SOCS3",
#         "SOX5",
#         "SPHK1",
#         "STXBP1",
#         "TF",
#         "TFAP2B",
#         "TFAP2C",
#         "TMEM18",
#         "TNFAIP3",
#         "TNNI3",
#         "TOMM20",
#         "TP53",
#         "TRIB3",
#         "TSLP",
#         "UCP2",
#         "UCP3",
#         "UBE2L3",
#         "UBQLN1",
#         "UQCRC1",
#         "VEGFA",
#         "VLDLR",
#         "WNT10B"
#     ]

def test_get_full_gene_nmf_for_gene_list():
    """
    Test that the gene nmf adaptare novelty function works.
    """
    # initialize
    map_result = {}
    p_value = 0.3

    # call method
    map_result = adapter.get_gene_full_nmf_for_gene_list(list_input_genes=list_gene_test, p_value_cutoff=p_value, log=True)

    # logger
    logger.info("got map result of size: {}".format(len(map_result.get('data'))))
    logger.info("got map result: {}".format(json.dumps(map_result.get('data'), indent=2)))

    # test
    assert map_result is not None
    assert map_result.get('data') is not None



