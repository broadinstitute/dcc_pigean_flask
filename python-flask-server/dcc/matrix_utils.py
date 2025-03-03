
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
import numpy as np
import copy
from scipy import sparse
from scipy.sparse import csc_matrix

import dcc.file_utils as futils 
import dcc.dcc_utils as dutils 

# constants
logger = dutils.get_logger(__name__)

# session vars
# matrix_gene_set = None
# map_gene_set_indexes_result = None
# if not matrix_gene_set:
#     map_gene_set, map_gene_set_indexes = load_geneset_set_matrix()



# methods
def load_geneset_matrix(map_gene_index, list_gene_set_files, path_gene_set_files, log=False):
    '''  
    will create the sparse matrix used for the pigean calculation
    '''
    # initialize
    list_row = []
    list_columns = []
    list_data = []
    count_column = 0
    data_value = 1
    matrix_result = None 
    map_gene_set_indexes_result = {}

    # read the file
    for file in list_gene_set_files:
        map_gene_set = futils.read_tab_delimited_file(path_gene_set_files + file)

        # log
        logger.info("read gene set file: {} with num entries: {}".format(file, len(map_gene_set)))

        # for each line, add 
        for gene_set_from_file, list_genes_from_file in map_gene_set.items():
            # log
            if log:
                logger.info("\nfor set: {}, got geneset list: {}".format(gene_set_from_file, list_genes_from_file))

            # for each gene, add data
            for gene in list_genes_from_file:
                if map_gene_index.get(gene) is not None:
                    list_row.append(map_gene_index.get(gene))
                    list_columns.append(count_column)
                    list_data.append(data_value)

                    # log
                    if log:
                        logger.info("adding gene: {} for row: {} and column: {}".format(gene, map_gene_index.get(gene), count_column))
                
                else:
                    if log:
                        logger.info("NOT adding gene: {} for row: {} and column: {}".format(gene, map_gene_index.get(gene), count_column))

        
            # add the gene set to the list
            map_gene_set_indexes_result[count_column] = gene_set_from_file

            # increment the column count for each gene set
            count_column = count_column + 1

    # log
    if log:
        logger.info("rows: {}".format(list_row))
        logger.info("cols: {}".format(list_columns))
        logger.info("data: {}".format(list_data))

        # print("\nrows: {}".format(list_row))
        # print("cols: {}".format(list_columns))
        # print("data: {}".format(list_data))

    # create the matrix
    # NOTE - remove toarray due to errors in compute_utils._calc_X_shift_scale() for .A1 and .power()
    # matrix_result = csc_matrix((list_data, (list_row, list_columns)), shape=(len(map_gene_index), len(map_gene_set_indexes_result))).toarray()
    matrix_result = csc_matrix((list_data, (list_row, list_columns)), shape=(len(map_gene_index), len(map_gene_set_indexes_result)))

    # log
    logger.info("returning gene set matrix of shape: {}".format(matrix_result.shape))
    
    # return
    return matrix_result, map_gene_set_indexes_result
    

def read_gene_phewas_bfs(genes, gene_to_ind, gene_phewas_bfs_in, min_value=1.0, **kwargs):

    #require X matrix

    if gene_phewas_bfs_in is None:
        logger.error("Require  gene-phewas-bfs-in file")

    logger.info("Reading --gene-phewas-bfs-in file %s" % gene_phewas_bfs_in)

    if gene_to_ind is None:
        logger.error("Need to initialixe --X before reading gene_phewas")

    Ys = None
    combineds = None
    priors = None

    row = []
    col = []
    with open(gene_phewas_bfs_in) as gene_phewas_bfs_fh:
        header_cols = gene_phewas_bfs_fh.readline().strip().split()

        pheno_col = 0
        if header_cols[pheno_col] != 'Trait':
            logger.warning("first column should be 'Trait' but is %s" % header_cols[1])

        id_col = 1
        if header_cols[id_col] != 'Gene':
            logger.warning("second column should be 'Gene' but is %s" % header_cols[0])

        combined_col = 2
        if header_cols[combined_col] != 'Combined':
            logger.warning("third column should be 'Combined' but is %s" % header_cols[2])

        bf_col = 3
        if header_cols[bf_col] != 'Direct':
            logger.warning("fourth column should be 'Direct' but is %s" % header_cols[3])

        prior_col = None

        if bf_col is not None:
            Ys  = []
        if combined_col is not None:
            combineds = []
        if prior_col is not None:
            priors = []

        phenos = []
        pheno_to_ind = {}

        num_gene_phewas_filtered = 0
        for line in gene_phewas_bfs_fh:
            cols = line.strip().split()
            if id_col >= len(cols) or pheno_col >= len(cols) or (bf_col is not None and bf_col >= len(cols)) or (combined_col is not None and combined_col >= len(cols)) or (prior_col is not None and prior_col >= len(cols)):
                logger.warning("Skipping due to too few columns in line: %s" % line)
                continue

            gene = cols[id_col]

            # if self.gene_label_map is not None and gene in self.gene_label_map:
            #     gene = self.gene_label_map[gene]

            if gene not in gene_to_ind:
                # logger.warning("Skipping gene %s not in gene index map" % gene)
                continue
            # logger.info("gene %s" % gene)

            pheno = cols[pheno_col]

            cur_combined = None
            if combined_col is not None:
                try:
                    combined = float(cols[combined_col])
                except ValueError:
                    if not cols[combined_col] == "NA":
                        logger.warning("Skipping unconvertible value %s for gene_set %s" % (cols[combined_col], gene))
                    continue

                if min_value is not None and combined < min_value:
                    num_gene_phewas_filtered += 1
                    continue

                cur_combined = combined

            if bf_col is not None:
                try:
                    bf = float(cols[bf_col])
                except ValueError:
                    if not cols[bf_col] == "NA":
                        logger.warning("Skipping unconvertible value %s for gene %s and pheno %s" % (cols[bf_col], gene, pheno))
                    continue

                if min_value is not None and combined_col is None and bf < min_value:
                    num_gene_phewas_filtered += 1
                    continue

                cur_Y = bf

            if prior_col is not None:
                try:
                    prior = float(cols[prior_col])
                except ValueError:
                    if not cols[prior_col] == "NA":
                        logger.warning("Skipping unconvertible value %s for gene_set %s" % (cols[prior_col], gene))
                    continue

                if min_value is not None and combined_col is None and bf_col is None and prior < min_value:
                    num_gene_phewas_filtered += 1
                    continue

                cur_prior = prior


            if pheno not in pheno_to_ind:
                pheno_to_ind[pheno] = len(phenos)
                phenos.append(pheno)

            pheno_ind = pheno_to_ind[pheno]

            if combineds is not None:
                combineds.append(cur_combined)
            if Ys is not None:
                Ys.append(cur_Y)
            if priors is not None:
                priors.append(cur_prior)

            col.append(pheno_ind)
            row.append(gene_to_ind[gene])

        #update what's stored internally
        # num_added_phenos = 0
        # if self.phenos is not None and len(self.phenos) < len(phenos):
        #     num_added_phenos = len(phenos) - len(self.phenos)

        # if num_added_phenos > 0:
        #     if self.X_phewas_beta is not None:
        #         self.X_phewas_beta = csc_matrix(sparse.vstack((self.X_phewas_beta, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta.shape[1])))))
        #     if self.X_phewas_beta_uncorrected is not None:
        #         self.X_phewas_beta_uncorrected = csc_matrix(sparse.vstack((self.X_phewas_beta_uncorrected, sparse.csc_matrix((num_added_phenos, self.X_phewas_beta_uncorrected.shape[1])))))

        # self.phenos = phenos
        # pheno_to_ind = self._construct_map_to_ind(phenos)

        #uniquify if needed
        row = np.array(row)
        col = np.array(col)
        indices = np.array(list(zip(row, col)))
        _, unique_indices = np.unique(indices, axis=0, return_index=True)
        if len(unique_indices) < len(row):
            logger.warning("Found %d duplicate values; ignoring duplicates" % (len(row) - len(unique_indices)))

        row = row[unique_indices]
        col = col[unique_indices]

        if combineds is not None:
            combineds = np.array(combineds)[unique_indices]
            gene_pheno_combined_prior_Ys = csc_matrix((combineds, (row, col)), shape=(len(genes), len(phenos)))

        if Ys is not None:
            Ys = np.array(Ys)[unique_indices]
            gene_pheno_Y = csc_matrix((Ys, (row, col)), shape=(len(genes), len(phenos)))

        if priors is not None:
            priors = np.array(priors)[unique_indices]
            gene_pheno_priors = csc_matrix((priors, (row, col)), shape=(len(genes), len(phenos)))
        
        # self.anchor_gene_mask = None
        # if anchor_genes is not None:
        #     self.anchor_gene_mask = np.array([x in anchor_genes for x in self.genes])
        #     if np.sum(self.anchor_gene_mask) == 0:
        #         logger.error("Couldn't find any match for %s" % list(anchor_genes))

        # self.anchor_pheno_mask = None
        # if anchor_phenos is not None:
        #     self.anchor_pheno_mask = np.array([x in anchor_phenos for x in self.phenos])
        #     if np.sum(self.anchor_pheno_mask) == 0:
        #         logger.error("Couldn't find any match for %s" % list(anchor_phenos))

        return phenos, gene_pheno_Y, gene_pheno_combined_prior_Ys#, gene_pheno_priors


def generate_gene_vector_from_list(list_gene, map_gene_index, log=False):
    '''
    will generate the gene vector from the list of genes
    '''
    # initialize
    vector_result = None 
    list_row = []
    list_columns = []
    list_data = []
    data_value = 1
    
    # from the genes, create a sparse vector
    for gene in list_gene:
        if map_gene_index.get(gene) is not None:
            # was getting (m, 1) vector, when wanted (1,m) vector
            # list_row.append(map_gene_index.get(gene))
            # list_columns.append(0)
            list_columns.append(map_gene_index.get(gene))
            list_row.append(0)
            list_data.append(data_value)

    # create the vector
    vector_result = csc_matrix((list_data, (list_row, list_columns)), shape=(1, len(map_gene_index))).toarray()

    # log
    if log:
        print("\nrows: {}".format(list_row))
        print("cols: {}".format(list_columns))
        print("data: {}".format(list_data))

    # return 
    return vector_result, list_columns

def sum_of_gene_row(sparse_matrix, gene_index, log=False):
    '''
    Function to compute the sum of all entries for a specific gene row
    '''
    return sparse_matrix[gene_index].sum()

# main
if __name__ == "__main__":
    pass