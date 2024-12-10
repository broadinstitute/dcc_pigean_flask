
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
import scipy
import scipy.sparse as sparse
import scipy.stats
from scipy.stats import truncnorm, norm
from scipy.special import psi as digamma
from scipy.special import erfc
import numpy as np
from numpy.random import gamma
from numpy.random import normal
from numpy.random import exponential
from sklearn.decomposition import NMF
import json
import time

import dcc.dcc_utils as dutils 
import dcc.matrix_utils as mutils 
import dcc.ml_utils as mlutils
from dcc.spike_n_slab import SnS


# TODO - process notes
# 1. done - X matrix is sitting in memory (something like 20K genes x 40K gene sets)
# 2. done - Request comes in with Y vector, need to reorder the Y vector to match the order of your X matrix
# 3. done - p_values = compute_beta_tildes (X, Y)
# 4. done - Filter V = X[:,p_values < 0.05] (edited) 
# 5. done - Then V = V[X.sum(axis=1) > 0,:] (edited) 
# 6. Then factor with V passed in as V0 (edited) (_bayes_nmf_l2())
# Y = np.zeros(X.shape[0])

# W is going to be gene x factor loadings
# K is gene set x factor loadings
# The factors are your groups
# The top scoring gene sets are your labels for the factors
# Then top scoring genes are the ones in the factor
# (Each factor is a cluster)

# constants
logger = dutils.get_logger(__name__)


BATCH_SIZE = 4500

class RunFactorException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# methods
def calculate_factors(matrix_gene_sets_gene_original, list_gene, list_system_genes, map_gene_index, map_gene_set_index, mean_shifts, scale_factors, 
                      p_value=0.05, max_num_gene_sets=100, is_factor_labels_llm=False, use_set_p_value=False, log=False):
    '''
    will produce the gene set factors and gene factors
    '''
    # initialize
    list_factor = [] 
    list_factor_genes = []
    list_factor_gene_sets = []
    gene_factor = None
    gene_set_factor = None
    map_factor_data_per_gene = {}
    logs_process = []

    # start time counter 
    start = time.time()

    # step 1/2: get the gene vector from the gene list
    if log:
        logger.info("step 0: got input gene list from user of size: {}".format(len(list_gene)))
    vector_gene, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_gene, map_gene_index=map_gene_index)

    # log
    if log:
        logger.info("step 1: got gene set matrix of shape: {}".format(matrix_gene_sets_gene_original.shape))
        logger.info("step 1: got mean_shifts of shape: {}".format(mean_shifts.shape))
        logger.info("step 1: got scale_factors of shape: {}".format(scale_factors.shape))
        logger.info("step 2: got one hot gene vector of shape: {}".format(vector_gene.shape))
        logger.info("step 2: got resulting found gene indices list of size: {}".format(len(list_input_gene_indices)))

    # step 3: get the p_values by gene set
    vector_gene_set_pvalues, _, _ = compute_beta_tildes(X=matrix_gene_sets_gene_original, Y=vector_gene, scale_factors=scale_factors, mean_shifts=mean_shifts)

    if log:
        logger.info("step 3: got p values vector of shape: {}".format(vector_gene_set_pvalues.shape))
        logger.info("step 3: filtering gene sets using p_value: {}".format(p_value))


    # NOTE - could add option to mulyily matrix by log(p_value)
    
    # step 4: filter the gene set columns based on computed pvalue for each gene set
    matrix_gene_set_filtered_by_pvalues, selected_gene_set_indices = filter_matrix_columns(matrix_input=matrix_gene_sets_gene_original, vector_input=vector_gene_set_pvalues, 
                                                                                           cutoff_input=p_value, max_num_gene_sets=max_num_gene_sets, log=log)
    # matrix_gene_set_filtered_by_pvalues, selected_gene_set_indices = filter_matrix_columns(matrix_input=matrix_gene_sets_gene_original, vector_input=vector_gene_set_pvalues, 
    #                                                                                        cutoff_input=0.5, log=log)

    if log:
        logger.info("step 4: got gene set filtered (col) matrix of shape: {}".format(matrix_gene_set_filtered_by_pvalues.shape))
        logger.info("step 4: got gene set filtered indices of length: {}".format(len(selected_gene_set_indices)))
        logger.info("step 4: got gene set filtered indices: {}".format(selected_gene_set_indices))

    # NOTE - only used in return at end
    # build filtered gene set list with p_values
    list_gene_set_p_values = build_gene_set_p_value_list(vector_gene_set_pvalues=vector_gene_set_pvalues, selected_gene_set_indices=selected_gene_set_indices, map_gene_set_index=map_gene_set_index)
    
    # step 5: filter gene rows by only the genes that are part of the remaining gene sets from the filtered gene set matrix
    matrix_gene_filtered_by_remaining_gene_sets, selected_gene_indices = filter_matrix_rows_by_sum_cutoff(matrix_to_filter=matrix_gene_set_filtered_by_pvalues, 
                                                                                                          matrix_to_sum=matrix_gene_set_filtered_by_pvalues, log=log)
    # check how many genes are left out
    list_input_genes_filtered_out_indices = [item for item in list_input_gene_indices if item not in selected_gene_indices.tolist()]    

    if log:
        logger.info("step 5: ===> got input gene filtered out of length: {}".format(len(list_input_genes_filtered_out_indices)))
        logger.info("step 5: got gene filtered indices of length: {}".format(len(selected_gene_indices)))
        logger.info("step 5: ===> got gene filtered (rows) matrix of shape: {} to start bayes NMF".format(matrix_gene_filtered_by_remaining_gene_sets.shape))
        # print("step 5: got gene filtered indices of length: {}".format(selected_gene_indices.shape))

    if not all(dim > 0 for dim in matrix_gene_filtered_by_remaining_gene_sets.shape):
        logger.info("step 6: ===> skipping rest of compute steps due to pre bayes NMF matrix of shape too small: {}".format(matrix_gene_filtered_by_remaining_gene_sets.shape))

    else:
        # TODO - this is the place to create new matrix to do bayes by

        
        # step 6: from this double filtered matrix, compute the factors
        gene_factor, gene_set_factor, _, _, exp_lambda, _ = _bayes_nmf_l2(V0=matrix_gene_filtered_by_remaining_gene_sets)
        # gene_factor, gene_set_factor = run_nmf(matrix_input=matrix_gene_filtered_by_remaining_gene_sets, log=log)

        if log:
            logger.info("step 6: got gene factor matrix of shape: {}".format(gene_factor.shape))
            logger.info("step 6: got gene set factor matrix of shape: {}".format(gene_set_factor.shape))
            logger.info("step 6: got lambda matrix of shape: {} with data: {}".format(exp_lambda.shape, exp_lambda))

        # step 7: find and rank the gene and gene set groups
        list_factor, list_factor_genes, list_factor_gene_sets, updated_gene_factors = rank_gene_and_gene_sets(X=None, Y=None, exp_lambdak=exp_lambda, exp_gene_factors=gene_factor, exp_gene_set_factors=gene_set_factor.T,
                                                                        list_system_genes=list_system_genes, map_gene_set_index=map_gene_set_index, 
                                                                        list_gene_mask=selected_gene_indices, list_gene_set_mask=selected_gene_set_indices, log=log)

        # step 7a - create the factor labels
        if is_factor_labels_llm:
            list_factor = mlutils.get_list_factor_names_from_llm(list_labels=list_factor, list_factor_gene_sets=list_factor_gene_sets)

        # step 7b - get the lowest factor per gene
        map_factor_data_per_gene = get_gene_factor_data_by_gene(exp_gene_factors=updated_gene_factors, list_system_genes=list_system_genes, 
                                                                  list_gene_mask=selected_gene_indices, list_factor_labels=list_factor, log=False)
        # print(json.dumps(map_lowest_factor_per_gene, indent=2))

        if log:
            logger.info("step 7: got factor list: {}".format(list_factor))
            logger.info("step 7: got gene list:")
            for row in list_factor_genes: 
                logger.info (row)
            logger.info("step 7: got gene set list:")
            for row in list_factor_gene_sets: 
                logger.info (row)

    # end time counter
    end = time.time()
    str_message = "compute process time is: {}s".format(end-start)
    logs_process.append(str_message)
    logs_process.append("used p_value: {}".format(p_value))
    logs_process.append("used max number of gene sets: {}".format(max_num_gene_sets))

    # log
    for row in logs_process:
        logger.info(row) 

    # only return the gene factors and gene set factors
    return list_factor, list_factor_genes, list_factor_gene_sets, gene_factor, gene_set_factor, map_factor_data_per_gene, list_gene_set_p_values, logs_process


def build_gene_set_p_value_list(vector_gene_set_pvalues, selected_gene_set_indices, map_gene_set_index, log=True):
    '''
    will build a sorted list of gene/p_value objects
    '''
    # initialize
    list_result = []

    # log
    if log:
        logger.info("P_VALUE_LIST - using p_value matrix of shape: {}".format(vector_gene_set_pvalues.shape))

    # build the list
    for index in selected_gene_set_indices:
        list_result.append({dutils.KEY_APP_GENE_SET: map_gene_set_index.get(index), dutils.KEY_APP_P_VALUE: vector_gene_set_pvalues[0, index]})

    # log
    if log:
        logger.info("P_VALUE_LIST - DATA: {}".format(list_result))

    # return
    return list_result


def group_factor_results(list_factor, list_factor_genes, list_factor_gene_sets, log=False):
    '''
    will group the results that come out of the computation steps
    '''
    list_result = []

    # loop through the factors
    for index, row in enumerate(list_factor):
        list_result.append({'top_set': row, 'gene_sets': list_factor_gene_sets[index], 'genes': list_factor_genes[index]})

    # return
    return list_result


def compute_beta_tildes(X, Y, scale_factors, mean_shifts, y_var=1, resid_correlation_matrix=None, log=False):
    '''
    get the scale factors and mean shifts from _calc_X_shift_scale()
    TODO - check if y_var default 1 is ok (Jason)
    '''

    logger.info("Calculating beta tildes")

    # Y can be a matrix with dimensions:
    # number of parallel runs x number of gene sets
    if len(Y.shape) == 2:
        len_Y = Y.shape[1]
        Y = (Y.T - np.mean(Y, axis=1)).T
    else:
        len_Y = Y.shape[0]
        Y = Y - np.mean(Y)

    # BUG - fixed y_var wrong setting
    y_var = np.var(Y, axis=1)
    
    dot_product = np.array(X.T.dot(Y.T) / len_Y).T

    variances = np.power(scale_factors, 2)

    # avoid divide by 0 only
    variances[variances == 0] = 1

    #multiply by scale factors because we store beta_tilde in units of scaled X
    beta_tildes = scale_factors * dot_product / variances

    if len(Y.shape) == 2:
        ses = np.outer(np.sqrt(y_var), scale_factors)
    else:
        ses = np.sqrt(y_var) * scale_factors

    ses /= (np.sqrt(variances * (len_Y - 1)))

    # FIXME: implement exact SEs
    # rather than just using y_var as a constant, calculate X.multiply(beta_tildes)
    # then, subtract out Y for non-zero entries, sum square, sum total
    # then, add in square of Y for zero entries, add in total
    # use these to calculate the variance term

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        logger.info("Adjusting standard errors for correlations")
        # need to multiply by inflation factors: (X * sigma * X) / variances

        # SEs and betas are stored in units of centered and scaled X
        # we do not need to scale X here, however, because cor_variances will then be in units of unscaled X
        # since variances are also in units of unscalAPOEed X, these will cancel out

        r_X = resid_correlation_matrix.dot(X)
        r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
        cor_variances = r_X_col_means - np.square(r_X_col_means)
        
        # never increase significance
        cor_variances[cor_variances < variances] = variances[cor_variances < variances]

        # both cor_variances and variances are in units of unscaled X
        se_inflation_factors = np.sqrt(cor_variances / variances)

    # MPD - change so only return pvalues
    # return finalize_regression(beta_tildes=beta_tildes, ses=ses, se_inflation_factors=se_inflation_factors)
    (cal_beta_tildes, cal_ses, cal_z_scores, cal_p_values, cal_se_inflation_factors) = finalize_regression(beta_tildes=beta_tildes, ses=ses, se_inflation_factors=se_inflation_factors)
    return cal_p_values, cal_beta_tildes, cal_ses

def finalize_regression(beta_tildes, ses, se_inflation_factors):

    if se_inflation_factors is not None:
        ses *= se_inflation_factors

    if np.prod(ses.shape) > 0:
        # empty mask
        empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
        max_se = np.max(ses)

        ses[empty_mask] = max_se * 100 if max_se > 0 else 100

        # if no y var, set beta tilde to 0

        beta_tildes[ses <= 0] = 0

    z_scores = np.zeros(beta_tildes.shape)
    ses_positive_mask = ses > 0
    z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
    if np.any(~ses_positive_mask):
        logger.info("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))

    p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))

    # pvalues is what I want
    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)



# NOTE - this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
# NOTE - paper is: Smith, K., Deutsch, A.J., McGrail, C. et al. Multi-ancestry polygenic mechanisms of type 2 diabetes. Nat Med 30, 1065–1074 (2024). https://doi.org/10.1038/s41591-024-02865-3
# 
# def _bayes_nmf_l2(V0, n_iter=10000, a0=10, tol=1e-7, K=15, K0=15, phi=1.0):
def _bayes_nmf_l2(V0, n_iter=10000, a0=10, tol=1e-3, K=15, K0=15, phi=1.0):
    '''
    example?
        result = _bayes_nmf_l2(matrix, a0=alpha0, K=max_num_factors, K0=max_num_factors)
        exp_lambdak = result[4]
        exp_gene_factors = result[1].T
        exp_gene_set_factors = result[0]
    '''
    # seed the random call
    np.random.seed(42)

    # convert the sparse matrix tto a dense array
    V0 = V0.toarray()

    # Bayesian NMF with half-normal priors for W and H
    # V0: input z-score matrix (variants x traits)
    # n_iter: Number of iterations for parameter optimization
    # a0: Hyper-parameter for inverse gamma prior on ARD relevance weights
    # tol: Tolerance for convergence of fitting procedure
    # K: Number of clusters to be initialized (algorithm may drive some to zero)
    # K0: Used for setting b0 (lambda prior hyper-parameter) -- should be equal to K
    # phi: Scaling parameter

    eps = 1.e-50
    delambda = 1.0
    #active_nodes = np.sum(V0, axis=0) != 0
    #V0 = V0[:,active_nodes]
    V = V0 - np.min(V0)
    Vmin = np.min(V)
    Vmax = np.max(V)
    N = V.shape[0]
    M = V.shape[1]

    W = np.random.random((N, K)) * Vmax #NxK
    H = np.random.random((K, M)) * Vmax #KxM

    I = np.ones((N, M)) #NxM
    V_ap = W.dot(H) + eps #NxM

    # print("1 V is: {} of dimension: {}".format(type(V), V.shape))
    temp = np.std(V)
    # print("2 V is: {} of dimension: {}".format(type(V), V.shape))
    temp = np.power(temp, 2) 
    # print("3 V is: {} of dimension: {}".format(type(V), V.shape))
    temp = temp * phi
    # print("4 V is: {} of dimension: {}".format(type(V), V.shape))

    phi = np.power(np.std(V), 2) * phi
    C = (N + M) / 2 + a0 + 1
    b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
    lambda_bound = b0 / C
    lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
    lambda_cut = lambda_bound * 1.5

    n_like = [None]
    n_evid = [None]
    n_error = [None]
    n_lambda = [lambdak]
    it = 1
    count = 1
    while delambda >= tol and it < n_iter:
        H = H * (W.T.dot(V)) / (W.T.dot(V_ap) + phi * H * np.repeat(1/lambdak, M).reshape(len(lambdak), M) + eps)
        V_ap = W.dot(H) + eps
        W = W * (V.dot(H.T)) / (V_ap.dot(H.T) + phi * W * np.tile(1/lambdak, N).reshape(N, len(lambdak)) + eps)
        V_ap = W.dot(H) + eps
        lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
        delambda = np.max(np.abs(lambdak - n_lambda[it - 1]) / n_lambda[it - 1])
        like = np.sum(np.power(V - V_ap, 2)) / 2
        n_like.append(like)
        n_evid.append(like + phi * np.sum((0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / lambdak + C * np.log(lambdak)))
        n_lambda.append(lambdak)
        n_error.append(np.sum(np.power(V - V_ap, 2)))
        if it % 100 == 0:
            logger.info("Iteration=%d; evid=%.3g; lik=%.3g; err=%.3g; delambda=%.3g; factors=%d; factors_non_zero=%d" % (it, n_evid[it], n_like[it], n_error[it], delambda, np.sum(np.sum(W, axis=0) != 0), np.sum(lambdak >= lambda_cut)))
        it += 1

    return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]
        #W # Variant weight matrix (N x K)
        #H # Trait weight matrix (K x M)
        #n_like # List of reconstruction errors (sum of squared errors / 2) pe_bayes_nmf_l2r iteration
        #n_evid # List of negative log-likelihoods per iteration
        #n_lambda # List of lambda vectors (shared weights for each of K clusters, some ~0) per iteration
        #n_error # List of reconstruction errors (sum of squared errors) per iteration

        # The key is that the lambdk (n_lambda) values tell you which factors remain vs. are zeroed out
        # And then genes (H) / gene sets (W) are assigned to the factors for which they have the highest weights
        # There is then logic to “score” each cluster and also annotate them with an LMM, but that is probably not necessary (yet)

def rank_gene_and_gene_sets(X, Y, exp_lambdak, exp_gene_factors, exp_gene_set_factors, list_gene_mask, list_gene_set_mask, list_system_genes, map_gene_set_index, cutoff=1e-5, log=False):
    '''
    will rank the gene sets and gene factors
    '''
    # self.exp_lambdak = result[4]
    # self.exp_gene_factors = result[1].T
    # self.exp_gene_set_factors = result[0]

    # log
    if log:
        logger.info("got lambda of shape: {}".format(exp_lambdak.shape))
        logger.info("got gene factor of shape: {}".format(exp_gene_factors.shape))
        logger.info("got gene set factor of shape: {}".format(exp_gene_set_factors.shape))

    # subset_down
    # GUESS: filter and keep if exp_lambdak > 0 and at least one non zero factor for a gene and gene set; then filter by cutoff
    factor_mask = exp_lambdak != 0 & (np.sum(exp_gene_factors, axis=0) > 0) & (np.sum(exp_gene_set_factors, axis=0) > 0)
    factor_mask = factor_mask & (np.max(exp_gene_set_factors, axis=0) > cutoff * np.max(exp_gene_set_factors))

    if log:
        logger.info("end up with factor mask of shape: {} and true count: {}".format(factor_mask.shape, np.sum(factor_mask)))

    # TODO - QUESTION
    # filter by factors; why invert factor_mask?
    if np.sum(~factor_mask) > 0:
        exp_lambdak = exp_lambdak[factor_mask]
        exp_gene_factors = exp_gene_factors[:,factor_mask]
        exp_gene_set_factors = exp_gene_set_factors[:,factor_mask]
    # elif self.betas_uncorrected is not None:
    #     gene_set_values = self.betas_uncorrected

    if log:
        logger.info("got NEW shrunk lambda of shape: {}".format(exp_lambdak.shape))
        logger.info("got NEW shrunk gene factor of shape: {}".format(exp_gene_factors.shape))
        logger.info("got NEW shrunk gene set factor of shape: {}".format(exp_gene_set_factors.shape))

    # gene_values = None
    # if self.combined_prior_Ys is not None:
    #     gene_values = self.combined_prior_Ys
    # elif self.priors is not None:
    #     gene_values = self.priors
    # elif self.Y is not None:
    #     gene_values = self.Y

    # if gene_set_values is not None:
    #     self.factor_gene_set_scores = self.exp_gene_set_factors.T.dot(gene_set_values[self.gene_set_factor_gene_set_mask])
    # else:
    #     self.factor_gene_set_scores = self.exp_lambdak

    factor_gene_set_scores = exp_lambdak

    # if gene_values is not None:
    #     self.factor_gene_scores = self.exp_gene_factors.T.dot(gene_values[self.gene_factor_gene_mask]) / self.exp_gene_factors.shape[0]
    # else:
    #     self.factor_gene_scores = self.exp_lambdak

    factor_gene_scores = exp_lambdak

    # get the indices with factors sorted in descending order
    reorder_inds = np.argsort(-factor_gene_set_scores)
    exp_lambdak = exp_lambdak[reorder_inds]

    factor_gene_set_scores = factor_gene_set_scores[reorder_inds]
    factor_gene_scores = factor_gene_scores[reorder_inds]
    exp_gene_factors = exp_gene_factors[:,reorder_inds]
    exp_gene_set_factors = exp_gene_set_factors[:,reorder_inds]

    #now label them
    # TODO - QUESTION - do I feed in the filtered ids here?
    # gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]
    # gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]

    # TODO - could make top count factors returned a variable; currently constant in code
    num_top = 100

    # get the top count for gene set and genes
    top_gene_inds = np.argsort(-exp_gene_factors, axis=0)[:num_top,:]               # not used
    top_gene_set_inds = np.argsort(-exp_gene_set_factors, axis=0)[:num_top,:]

    factor_labels = []
    top_gene_sets = []
    top_genes = []
    factor_prompts = []

    # log
    if log:
        logger.info("looping through factor gene set scores of size: {} and data: \n{}".format(len(factor_gene_set_scores), factor_gene_set_scores))
        logger.info("got top pathway ids type: {} and data: {}".format(type(top_gene_set_inds), top_gene_set_inds))
        logger.info("got top gene ids: {}".format(top_gene_inds))

    for i in range(len(factor_gene_set_scores)):
        # orginal for reference
        # top_gene_sets.append([self.gene_sets[i] for i in np.where(self.gene_set_factor_gene_set_mask)[0][top_gene_set_inds[:,i]]])
        # top_genes.append([self.genes[i] for i in np.where(self.gene_factor_gene_mask)[0][top_gene_inds[:,i]]])
        
        # list_temp = top_gene_inds[:,i].tolist()
        # print("top gene ids: {}".format(list_temp))

        # build the list of genes and gene sets that were filtered first by p_value, then the factor process in this method
        list_gene_index_factor = get_referenced_list_elements(list_referenced=list_gene_mask, list_index=top_gene_inds[:,i].tolist(), log=False)
        list_gene_set_index_factor = get_referenced_list_elements(list_referenced=list_gene_set_mask, list_index=top_gene_set_inds[:,i].tolist(), log=False)

        # print("map type: {}".format(type(map_gene_set_index)))
        # print("list type: {}".format(type(list_gene_set_index_factor)))
        # print("list: {}".format(list_gene_set_index_factor))

        # if log:
        #     print("got pathway indexes: {}".format(list_gene_set_index_factor))
        #     print("got gene indexes: {}".format(list_gene_index_factor))

        # build the list of groupings of gene sets
        list_temp = []
        for index_local, gs_index in enumerate(list_gene_set_index_factor):
            score_gene = exp_gene_set_factors[top_gene_set_inds[index_local, i], i]
            if score_gene > 0.01:
                list_temp.append({'gene_set': map_gene_set_index.get(gs_index), 'score': score_gene})
        top_gene_sets.append(list_temp)
        # top_gene_sets.append([map_gene_set_index.get(gs_index) for gs_index in list_gene_set_index_factor])

        # build the list of groupings of genes (cutoff of 0.01)
        # top_genes.append([list_system_genes[g_index] for g_index in list_gene_index_factor])
        # top_genes.append([list_system_genes[g_index] for index_local, g_index in enumerate(list_gene_index_factor) if exp_gene_factors[top_gene_inds[index_local], i] > 0.01])
 
        # adding genes and their scores to the factor list
        list_temp = []
        for index_local, g_index in enumerate(list_gene_index_factor):
            score_gene = exp_gene_factors[top_gene_inds[index_local, i], i]
            if score_gene > 0.01:
                list_temp.append({'gene': list_system_genes[g_index], 'score': score_gene})
        top_genes.append(list_temp)

        factor_labels.append(top_gene_sets[i][0] if len(top_gene_sets[i]) > 0 else {'gene_set': None, 'score': None})

        # build the prompt
        # factor_prompts.append(",".join(item['gene_set'] for item in top_gene_sets[i]))
        # ';'.join([item['gene'] for item in list_factor_genes[index]])
        # factor_prompts.append(",".join(top_gene_sets[i]))

    if log:
        logger.info("got factor labels of size: {} and data: {}".format(len(factor_labels), factor_labels))

    # return the 3 grouping data structures, then the updated (filtered) gene factors
    return factor_labels, top_genes, top_gene_sets, exp_gene_factors


def get_gene_factor_data_by_gene(exp_gene_factors, list_system_genes, list_gene_mask, list_factor_labels, log=False):
    '''
    will return the lowest factor per gene - to be used as novelty calculation for ARS
    '''
    # initialize
    map_result = {}

    if all(dim > 0 for dim in exp_gene_factors.shape):    
        # log
        if log:
            logger.info("lowest factor - got gene factor of shape: {}".format(exp_gene_factors.shape))
            # print("lowest factor - got filtered gene mask of size: {} and data: \n{}".format(len(list_gene_mask), list_gene_mask))

        # get the lowest value per row
        # BUG - fixed for best factor per gene
        min_per_row = np.max(exp_gene_factors, axis=1)

        if log:
            logger.info("lowest factor - got gene factor MINIMUM of shape: {} and type: {}".format(min_per_row.shape, type(min_per_row)))
            for index in range(len(list_gene_mask)):
                logger.info("lowest factor - for gene: {} get factor : {}".format(list_system_genes[list_gene_mask[index]], exp_gene_factors[index]))

        # build the map
        if min_per_row is not None:
            for index, row_factor in enumerate(min_per_row.tolist()):
                index_system = list_gene_mask[index]
                gene_name = list_system_genes[index_system]
                map_result[gene_name] = {dutils.KEY_INTERNAL_LOWEST_FACTOR_SCORE: row_factor}


        # for each gene, get the factor with the highest score
        top_factor_indices_per_gene = np.argmax(exp_gene_factors, axis=1)
        if top_factor_indices_per_gene is not None:
            for index, row_factor_index in enumerate(top_factor_indices_per_gene.tolist()):
                index_system = list_gene_mask[index]
                gene_name = list_system_genes[index_system]
                map_result[gene_name][dutils.KEY_INTERNAL_HIGHEST_FACTOR_NAME] = list_factor_labels[row_factor_index]
                map_result[gene_name][dutils.KEY_INTERNAL_HIGHEST_FACTOR_SCORE] = exp_gene_factors[index, row_factor_index]


        # log
        logger.info("data: {}".format(map_result.get('APOE')))
        logger.info("returning lowest factor (novelty) and other data gene map of size: {}".format(len(map_result)))

    else:
        logger.info("returning empty map for exp_gene_factor of shape: {}".format(exp_gene_factors.shape))


    # return
    return map_result


def get_referenced_list_elements(list_referenced, list_index, log=False):
    '''
    will return a list of the referenced elements from the list
    '''
    list_result = []

    # log
    if log:
        logger.info("ref list: {}".format(list_referenced))
        logger.info("index list: {}".format(list_index))

    # get the elements
    list_result = [list_referenced[i] for i in list_index]

    # return
    return list_result



def _calc_X_shift_scale(X, y_corr_cholesky=None):
    '''
    returns the mean shifts and scale factors for the initial pathwayxgene matrix X
    '''
    if y_corr_cholesky is None:
        mean_shifts = X.sum(axis=0).A1 / X.shape[0]
        scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))

        # NOTE - delete? mods not necessary if don't convert cscsparse.toarray()
        # mean_shifts = np.ravel(X.sum(axis=0)) / X.shape[0]
        # scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))
    else:
        scale_factors = np.array([])
        mean_shifts = np.array([])
        for X_b, begin, end, batch in _get_X_blocks_internal(X, y_corr_cholesky):
            (cur_mean_shifts, cur_scale_factors) = _calc_shift_scale(X_b)
            mean_shifts = np.append(mean_shifts, cur_mean_shifts)
            scale_factors = np.append(scale_factors, cur_scale_factors)

    # log
    logger.info("returning mean shifts matrix of shape: {}".format(mean_shifts.shape))
    logger.info("returning scale factors matrix of shape: {}".format(scale_factors.shape))

    return (mean_shifts, scale_factors)


def _calc_shift_scale(X_b):
    '''
    used in _calc_X_shift_scale()
    '''
    mean_shifts = []
    scale_factors = []
    for i in range(X_b.shape[1]):
        X_i = X_b[:,i]
        mean_shifts.append(np.mean(X_i))
        scale_factor = np.std(X_i)
        if scale_factor == 0:
            scale_factor = 1
        scale_factors.append(scale_factor)
    return (np.array(mean_shifts), np.array(scale_factors))


def _get_X_blocks_internal(X_orig, y_corr_cholesky, whiten=True, full_whiten=False, start_batch=0, mean_shifts=None, scale_factors=None):
    ''' 
    used in _calc_X_shift_scale()
    '''

    if y_corr_cholesky is None:
        #explicitly turn these off to help with caching
        whiten = False
        full_whiten = False

    num_batches = _get_num_X_blocks(X_orig)

    consider_cache = X_orig is X_orig and num_batches == 1 and mean_shifts is None and scale_factors is None

    for batch in range(start_batch, num_batches):
        # logger.info("Getting X%s block batch %s (%s)" % ("_missing" if X_orig is X_orig_missing_gene_sets else "", batch, "fully whitened" if full_whiten else ("whitened" if whiten else "original")), TRACE)
        begin = batch * BATCH_SIZE
        end = (batch + 1) * BATCH_SIZE
        if end > X_orig.shape[1]:
            end = X_orig.shape[1]

        if last_X_block is not None and consider_cache and last_X_block[1:] == (whiten, full_whiten, begin, end, batch):
            logger.info("Using cache!")
            yield (last_X_block[0], begin, end, batch)
        else:
            X_b = X_orig[:,begin:end].toarray()
            if mean_shifts is not None:
                X_b = X_b - mean_shifts[begin:end]
            if scale_factors is not None:
                X_b = X_b / scale_factors[begin:end]

            if whiten or full_whiten:
                X_b = _whiten(X_b, y_corr_cholesky, whiten=whiten, full_whiten=full_whiten)

            #only cache if we are accessing the original X
            if consider_cache:
                last_X_block = (X_b, whiten, full_whiten, begin, end, batch)
            else:
                last_X_block = None

            yield (X_b, begin, end, batch)


def _whiten(matrix, corr_cholesky, whiten=True, full_whiten=False):
    '''
    used in _get_X_blocks_internal()
    '''
    if full_whiten:
        #fully whiten, by sigma^{-1}; useful for optimization
        matrix = scipy.linalg.cho_solve_banded((corr_cholesky, True), matrix, overwrite_b=True)
    elif whiten:
        #whiten X_b by sigma^{-1/2}
        matrix = scipy.linalg.solve_banded((corr_cholesky.shape[0]-1, 0), corr_cholesky, matrix, overwrite_ab=True)
    return matrix


def _get_num_X_blocks(X_orig, batch_size=None):
    '''
    used in _get_X_blocks_internal()
    '''
    if batch_size is None:
        batch_size = BATCH_SIZE
    return int(np.ceil(X_orig.shape[1] / batch_size))


def filter_matrix_columns(matrix_input, vector_input, cutoff_input, max_num_gene_sets, log=False):
    '''
    will filter the matrix based on the vector and cutoff
    the columns are gene sets in this instance
    '''

    # REFERENCE
    # keep_metrics = np.where(np.any(metric_p_values_m < 0.05, axis=0))[0]

    # this works
    # selected_column_indices = np.where(np.any(vector_input < cutoff_input, axis=0))[0]
    # TODO this does not
    # selected_column_indices = np.where(mask)[0]

    # log
    if log:
        logger.info("got matrix to filter of shape: {} and type: {}".format(matrix_input.shape, type(matrix_input)))
        logger.info("got filter vector of shape: {} and type: {}".format(vector_input.shape, type(vector_input)))
        # logger.info("passing vector value: {}".format(vector_input[0,51864]))

    # select the columns that pass the p_value cutoff
    selected_column_indices = np.where(np.any(vector_input < cutoff_input, axis=0))[0]

    # CHECK - if there are more selected columns than the max_column parameter, take the top columns only
    if len(selected_column_indices) > max_num_gene_sets:
        # log
        if log:
            logger.info("filtered gene sets of size: {} is LARGER than the max: {}, so taking top {}".format(len(selected_column_indices), max_num_gene_sets, max_num_gene_sets))

        # Get the indices of the n lowest values
        min_values = np.min(vector_input, axis=0)
        selected_column_indices = np.argsort(min_values)[:max_num_gene_sets]
    else:
        if log:
            logger.info("filtered gene sets of size: {} is SMALLER than the max: {}, so keep the result as is".format(len(selected_column_indices), max_num_gene_sets))


    # filter the reference gene/gene sets matrix down
    matrix_result = matrix_input[:, selected_column_indices]

    # log
    if log:
        logger.info("vector values that passed {} filter or are top {} gene sets: {}".format(cutoff_input, max_num_gene_sets, vector_input[0, selected_column_indices]))
        logger.info("got filtered column list of length: {}".format(len(selected_column_indices)))
        logger.info("got filtered column list of: {}".format(selected_column_indices))
        logger.info("got resulting shape of column filters from: {} to {}".format(matrix_input.shape, matrix_result.shape))
        # logger.info("filtered matrix: {}".format(matrix_result))

    # return
    return matrix_result, selected_column_indices


def filter_matrix_rows_by_sum_cutoff(matrix_to_filter, matrix_to_sum, cutoff_input=0, log=False):
    '''
    will filter the matrix based on sum of each row and cutoff
    '''
    # mask = matrix_to_sum.sum(axis=1) > cutoff_input
    # selected_indices = np.where(mask)[0]
    # keep_metrics = np.where(np.any(mask, axis=0))[0]
    # matrix_result = matrix_to_filter[keep_metrics, :]
    # # matrix_result = matrix_to_filter[mask, :]

    if log:
        logger.info("got matrix to filter of shape: {} and type: {}".format(matrix_to_filter.shape, type(matrix_to_filter)))
        logger.info("got matrix to sum of shape: {} and type: {}".format(matrix_to_sum.shape, type(matrix_to_sum)))

    mask = matrix_to_sum.sum(axis=1) > cutoff_input
    # selected_indices = np.where(mask)[0]
    selected_indices = np.where(np.any(mask, axis=1))[0]
    matrix_result = matrix_to_filter[selected_indices, :]
    # matrix_result = matrix_to_filter[mask, :]

    # log
    if log:
        logger.info("got resulting shape from row sum filters from: {} to {}".format(matrix_to_filter.shape, matrix_result.shape))
        # print("got filter rows indices: {}".format(selected_indices))
        # print("example matrix to sum: {}".format(matrix_to_sum.toarray()[2]))

    # return
    return matrix_result, selected_indices


def run_nmf(matrix_input, num_components=15, log=False):
    '''
    run the sklearn NMF
    '''
    # Initialize the NMF model
    model = NMF(n_components=num_components, random_state=42)

    # Fit the model and transform the matrix X
    W = model.fit_transform(matrix_input)
    H = model.components_    

    # log
    if log:
        logger.info("for gene factor of shape: {}".format(W.shape))
        logger.info("for gene set factor of shape: {}".format(H.shape))

    # return
    return W, H


# def calculate_gene_scores_map(matrix_gene_sets, list_input_genes, map_gene_index, map_gene_set_index, mean_shifts, scale_factors, log=False):
# def calculate_gene_scores_map(matrix_gene_sets, list_input_genes, map_gene_index, list_system_genes, input_p_values, input_beta_tildes, input_ses, log=False):
#     '''
#     calculates the gene scores
#     '''
#     # initialize
#     map_gene_scores = {}

#     # get the matrix/vectors needed
#     vector_gene, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_input_genes, map_gene_index=map_gene_index)

#     # log
#     logger.info("form a gene list of size: {} got a gene index of size: {} and gene vector of shape: {}".format(len(list_input_genes), len(list_input_gene_indices), vector_gene.shape))
               
#     # convert data for Alex's code
#     # make sure gene set matrix is full matrix, not sparse
#     # TODO transpose matrices to get genes with gene sets as features (gene rows by gen set columns)
#     matrix_dense_gene_sets = matrix_gene_sets.toarray()

#     # get the coeff
#     start = time.time()
#     mod_log_sns = SnS()


#     # TODO - get betas and standard errors
#     # cal_beta_tildes, cal_ses from first pvalue calc - gene set specific
#     # 

#     # get the new betas, ses
#     # need to use dense marix; error with sparse matrix on .std() call
#     # logger.info("calculating beta tildes for gene scores")
#     # # TODO - not needed 
#     # log_coeff_beta_tildes, log_coeff_ses, _, log_coeff_pvalue, _, _, _ = mod_log_sns.compute_logistic_beta_tildes(X=matrix_dense_gene_sets, Y=vector_gene)

#     # log
#     # logger.info("got beta tildes shape: {}".format(log_coeff_beta_tildes.shape))
#     logger.info("got beta tildes shape: {}".format(input_beta_tildes.shape))

#     # TODO - try using sparse matrix here for speed
#     # step to calculate correlation and filter out gene set features
#     # filter on pvalue
#     # similar to runnagene prining; not needed here
#     prune_val = 0.8
#     # features_to_keep = mod_log_sns.prune_gene_sets(matrix_dense_gene_sets, log_coeff_pvalue, prune_value=prune_val)
#     features_to_keep = np.zeros(matrix_dense_gene_sets.shape[1], dtype=bool)
#     features_to_keep[0:100]=True
#     logger.info("got filter of shape: {} and data: {}".format(features_to_keep.shape, features_to_keep))

#     # get the gene scores
#     logger.info("calculating new betas for gene scores")
#     # gene_betas, _ = mod_log_sns.calculate_non_inf_betas(
#     #                 log_coeff_beta_tildes[:, list_input_gene_indices],
#     #                 log_coeff_ses[:, list_input_gene_indices],
#     #                 X_orig=matrix_dense_gene_sets[:, list_input_gene_indices],
#     #                 sigma2=np.ones((1, 1)),
#     #                 p=np.ones((1, 1)) * 0.001)

#     # TODO - try with the sparse matrix for speed
#     # if verytthing is 0, increase the variance (increasr by factor of 10)
#     variance = 0.001
#     # TODO - jf - cut on .05 pvalue (reduce gene sets)
#     # TODO this will be gene set scores (this is beta; effect of gene set of whether gene is in gene set)
#     gene_set_betas, _ = mod_log_sns.calculate_non_inf_betas(
#                     log_coeff_beta_tildes[features_to_keep],
#                     log_coeff_ses[features_to_keep],
#                     X_orig=matrix_dense_gene_sets[:, features_to_keep],
#                     sigma2=np.ones((1, 1)) * variance,
#                     p=np.ones((1, 1)) * 0.001)

#     # TODO - naive_priors
#     # input betas above and x original matrix; also scale factors
#     # TODO - look at naive priors adjusted function
#     # return all inpout genes and score and extra genes with high scores

#     # old alex coded
#     # gene_betas, _ = mod_log_sns.calculate_non_inf_betas(
#     #                 log_coeff_beta_tildes[:, features_to_keep],
#     #                 log_coeff_ses[:, features_to_keep],
#     #                 X_orig=matrix_dense_gene_sets[:, features_to_keep],
#     #                 sigma2=np.ones((1, 1)) * variance,
#     #                 p=np.ones((1, 1)) * 0.001)

#     # log
#     end = time.time()
#     str_message = "gene scores calculation time elapsed {}s".format(end-start)
#     logger.info(str_message)
#     logger.info("got gene scores of shape: {}: and data: {}".format(gene_betas.shape, gene_betas))

#     # build the map
#     for index, gene_score in enumerate(gene_betas):
#         index_gene = list_input_gene_indices[index]
#         map_gene_scores[list_system_genes[index_gene]] = gene_score

#     # return
#     return map_gene_scores

def calculate_gene_scores_map(matrix_gene_sets, list_input_genes, map_gene_index, map_gene_set_index, list_system_genes, input_mean_shifts, input_scale_factors, log=False):
    '''
    calculates the gene scores
    '''
    # initialize
    map_gene_scores = {}
    map_gene_set_scores = {}
    logs_process = []

    # get the matrix/vectors needed
    vector_gene, list_input_gene_indices = mutils.generate_gene_vector_from_list(list_gene=list_input_genes, map_gene_index=map_gene_index)

    # log
    logger.info("from a gene list of size: {} got a gene index of size: {} and gene vector of shape: {}".format(len(list_input_genes), len(list_input_gene_indices), vector_gene.shape))
               
    # convert data for Alex's code
    # make sure gene set matrix is full matrix, not sparse
    # TODO transpose matrices to get genes with gene sets as features (gene rows by gen set columns)
    # matrix_dense_gene_sets = matrix_gene_sets.toarray()

    # get the coeff
    start = time.time()
    mod_log_sns = SnS()

    # TODO - get betas and standard errors
    # DONE - comes in from input
    # cal_beta_tildes, cal_ses from first pvalue calc - gene set specific
    logger.info("gene scores - calculating p_values for gene scores")
    vector_gene_set_pvalues, vector_beta_tildes, vector_ses = compute_beta_tildes(X=matrix_gene_sets, Y=vector_gene, scale_factors=input_scale_factors, mean_shifts=input_mean_shifts)

    # log
    logger.info("got calculated beta tildes shape: {}".format(vector_beta_tildes.shape))
    logger.info("got calculated ses shape: {}".format(vector_ses.shape))
    logger.info("got input scale factors shape: {}".format(input_scale_factors.shape))
    logger.info("got calculated p_values shape: {}".format(vector_gene_set_pvalues.shape))

    # filter the gene set columns based on computed pvalue for each gene set
    # TODO - jf - cut on .05 pvalue (reduce gene sets)
    max_gene_sets = 500
    max_gene_sets = 200
    matrix_gene_set_filtered_by_pvalues, selected_gene_set_indices = filter_matrix_columns(matrix_input=matrix_gene_sets, vector_input=vector_gene_set_pvalues, 
                                                                                           cutoff_input=0.05, max_num_gene_sets=max_gene_sets, log=log)
    # matrix_gene_set_filtered_by_pvalues, selected_gene_set_indices = filter_matrix_columns(matrix_input=matrix_gene_sets_gene_original, vector_input=vector_gene_set_pvalues, 
    #                                                                                        cutoff_input=0.5, log=log)

    logger.info("got gene set filtered (row) matrix of shape: {}".format(matrix_gene_set_filtered_by_pvalues.shape))
    logger.info("got gene set filtered indices of length: {}".format(len(selected_gene_set_indices)))
    # logger.info("got gene set filtered indices: {}".format(selected_gene_set_indices))

    # NOTE - filter genes for speed
    # filter gene rows by only the genes that are part of the remaining gene sets from the filtered gene set matrix
    matrix_gene_filtered_by_remaining_gene_sets, selected_gene_indices = filter_matrix_rows_by_sum_cutoff(matrix_to_filter=matrix_gene_set_filtered_by_pvalues, 
                                                                                                          matrix_to_sum=matrix_gene_set_filtered_by_pvalues, log=log)
    logger.info("got gene filtered (col) matrix of shape: {}".format(matrix_gene_filtered_by_remaining_gene_sets.shape))
    logger.info("got gene filtered indices of length: {}".format(len(selected_gene_indices)))


    # get the gene scores
    logger.info("calculating gene set betas for gene scores")
    # if verytthing is 0, increase the variance (increasr by factor of 10)
    variance = 0.001
    # TODO this will be gene set scores (this is beta; effect of gene set of whether gene is in gene set)

    # NOTE - too long for full matrix
    # gene_set_betas, _ = mod_log_sns.calculate_non_inf_betas(
    #                 input_beta_tildes,
    #                 input_ses,
    #                 X_orig=matrix_dense_gene_sets,
    #                 sigma2=np.ones((1, 1)) * variance,
    #                 p=np.ones((1, 1)) * 0.001)
    filtered_beta_tildes = vector_beta_tildes[:, selected_gene_set_indices]
    filtered_ses = vector_ses[:, selected_gene_set_indices]
    gene_set_betas, _ = mod_log_sns.calculate_non_inf_betas(
                    filtered_beta_tildes,
                    filtered_ses,
                    X_orig=matrix_gene_filtered_by_remaining_gene_sets.toarray(),
                    sigma2=np.ones((1, 1)) * variance,
                    p=np.ones((1, 1)) * 0.001)
    
    logger.info("gene scores: got gene set betas of shape: {}".format(gene_set_betas.shape))
    # logger.info("gene scores: got gene set betas of data: {}".format(gene_set_betas))

    # gene_set_betas, _ = mod_log_sns.calculate_non_inf_betas(
    #                 filtered_beta_tildes,
    #                 filtered_ses,
    #                 X_orig=matrix_dense_gene_sets[selected_gene_set_indices, :],
    #                 sigma2=np.ones((1, 1)) * variance,
    #                 p=np.ones((1, 1)) * 0.001)
    # gene_set_betas, _ = mod_log_sns.calculate_non_inf_betas(
    #                 log_coeff_beta_tildes[features_to_keep],
    #                 log_coeff_ses[features_to_keep],
    #                 X_orig=matrix_dense_gene_sets[:, features_to_keep],
    #                 sigma2=np.ones((1, 1)) * variance,
    #                 p=np.ones((1, 1)) * 0.001)

    # TODO - look at naive priors adjusted function
    # input betas above and x original matrix; also scale factors
    # get the priors
    filtered_scale_factors = input_scale_factors[selected_gene_set_indices]
    result_priors = calculate_naive_priors(input_matrix_gene_set=matrix_gene_filtered_by_remaining_gene_sets, 
        input_vector_genes=vector_gene, input_betas=gene_set_betas, input_scale_factors=filtered_scale_factors, log=log)

    # log
    logger.info("gene scores: got result gene naive priors of shape: {}".format(result_priors.shape))
    logger.info("gene scores: got result gene naive priors of data: {}".format(result_priors))

    # log
    end = time.time()
    str_message = "gene scores calculation time elapsed {}s".format(end-start)
    logger.info(str_message)
    # logger.info("got gene scores of shape: {}: and data: {}".format(gene_betas.shape, gene_betas))

    # build the gene set map
    for index, gene_set_score in enumerate(gene_set_betas):
        index_gene_set = selected_gene_set_indices[index]
        map_gene_set_scores[map_gene_set_index[index_gene_set]] = gene_set_score

    # build the gene map
    # return all input genes and score and extra genes with high scores
    for index, gene_score in enumerate(result_priors):
        index_gene = selected_gene_indices[index]
        map_gene_scores[list_system_genes[index_gene]] = gene_score

    # return
    return map_gene_scores, map_gene_set_scores, logs_process


def calculate_naive_priors(input_matrix_gene_set, input_vector_genes, input_betas, input_scale_factors, log=False):
    '''
    calculate the gene scores based on the input gene set betas
    '''
    #  initialize

    # log
    logger.info("got input gene set matrix of shape: {}".format(input_matrix_gene_set.shape))
    logger.info("got input gene vector of shape: {}".format(input_vector_genes.shape))
    logger.info("got input betas of shape: {}".format(input_betas.shape))
    logger.info("got input scale factors of shape: {}".format(input_scale_factors.shape))

    # get the priors
    result_priors = input_matrix_gene_set.dot(input_betas / input_scale_factors)
    result_priors_missing = np.array([])

    # log
    if log:
        logger.info("got priors of shape: {}".format(result_priors.shape))

    # get the mean
    total_mean = np.mean(np.concatenate((result_priors, result_priors_missing)))
    result_priors -= total_mean
    result_priors_missing -= total_mean

    # self.calculate_priors_adj()

    # # TODO - is this necessary with one hot vector?
    # if self.Y is not None:
    #     if self.priors is not None:
    #         self.combined_prior_Ys = self.priors + self.Y
    #     if self.priors_adj is not None:
    #         self.combined_prior_Ys_adj = self.priors_adj + self.Y

    # TODO - what do I return?
    return result_priors


def calculate_priors_adj(input_matrix_gene_sets, input_priors, log=False):
    #do the regression
    # gene_N = self.get_gene_N()

    # gene_N_missing = self.get_gene_N(get_missing=True)
    all_gene_N = gene_N
    # if self.genes_missing is not None:
    #     assert(gene_N_missing is not None)
    #     all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

    # if self.genes_missing is not None:
    #     total_priors = np.concatenate((self.priors, self.priors_missing))
    # else:
    #     total_priors = self.priors
    # total_priors = self.priors
    total_priors = input_priors

    priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
    priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

    logger.info("Adjusting priors with slope %.4g" % priors_slope)
    # priors_adj = self.priors - priors_slope * gene_N - priors_intercept
    priors_adj = input_priors - priors_slope * gene_N - priors_intercept

    # if overwrite_priors:
    #     self.priors = priors_adj
    # else:
    #     self.priors_adj = priors_adj
    # if self.genes_missing is not None:
    #     priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing
    #     if overwrite_priors:
    #         self.priors_missing = priors_adj_missing
    #     else:
    #         self.priors_adj_missing = priors_adj_missing

    # return
    return priors_adj


# main
if __name__ == "__main__":
    pass








# old
# 20241015 - claculate gene scores

# def calculate_naive_priors(self, adjust_priors=False):
#     if self.X_orig is None:
#         bail("X is required for this operation")
#     if self.betas is None:
#         bail("betas are required for this operation")
    
#     self.priors = self.X_orig.dot(self.betas / self.scale_factors)

#     if self.X_orig_missing_genes is not None:
#         self.priors_missing = self.X_orig_missing_genes.dot(self.betas / self.scale_factors)
#     else:
#         self.priors_missing = np.array([])

#     total_mean = np.mean(np.concatenate((self.priors, self.priors_missing)))
#     self.priors -= total_mean
#     self.priors_missing -= total_mean

#     self.calculate_priors_adj(overwrite_priors=adjust_priors)

#     if self.Y is not None:
#         if self.priors is not None:
#             self.combined_prior_Ys = self.priors + self.Y
#         if self.priors_adj is not None:
#             self.combined_prior_Ys_adj = self.priors_adj + self.Y

# def get_gene_N(self, get_missing=False):
#     if get_missing:
#         if self.gene_N_missing is None:
#             return None
#         else:
#             return self.gene_N_missing + (self.gene_ignored_N_missing if self.gene_ignored_N_missing is not None else 0)
#     else:
#         if self.gene_N is None:
#             return None
#         else:
#             return self.gene_N + (self.gene_ignored_N if self.gene_ignored_N is not None else 0)


# def calculate_priors_adj(self, overwrite_priors=False):
#     if self.priors is None:
#         return
    
#     #do the regression
#     gene_N = self.get_gene_N()
#     gene_N_missing = self.get_gene_N(get_missing=True)
#     all_gene_N = gene_N
#     if self.genes_missing is not None:
#         assert(gene_N_missing is not None)
#         all_gene_N = np.concatenate((all_gene_N, gene_N_missing))

#     if self.genes_missing is not None:
#         total_priors = np.concatenate((self.priors, self.priors_missing))
#     else:
#         total_priors = self.priors

#     priors_slope = np.cov(total_priors, all_gene_N)[0,1] / np.var(all_gene_N)
#     priors_intercept = np.mean(total_priors - all_gene_N * priors_slope)

#     log("Adjusting priors with slope %.4g" % priors_slope)
#     priors_adj = self.priors - priors_slope * gene_N - priors_intercept
#     if overwrite_priors:
#         self.priors = priors_adj
#     else:
#         self.priors_adj = priors_adj
#     if self.genes_missing is not None:
#         priors_adj_missing = self.priors_missing - priors_slope * gene_N_missing
#         if overwrite_priors:
#             self.priors_missing = priors_adj_missing
#         else:
#             self.priors_adj_missing = priors_adj_missing



    #there are two levels of parallelization here:
    #1. num_chains: sample multiple independent chains with the same beta/se/V
    #2. multiple parallel runs with different beta/se (and potentially V). To do this, pass in lists of beta and se (must be the same length) and an optional list of V (V must have same length as beta OR must be not a list, in which case the same V will be used for all betas and ses

    #to run this in parallel, pass in two-dimensional matrix for beta_tildes (rows are parallel runs, columns are beta_tildes)
    #you can pass in multiple V as well with rows/columns mapping to gene sets and a first dimension mapping to parallel runs
# def _calculate_non_inf_betas(self, initial_p, return_sample=False, max_num_burn_in=None, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, eps=0.01, max_frac_sem=0.01, max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False, sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False, sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None, beta_tildes=None, ses=None, V=None, X_orig=None, scale_factors=None, mean_shifts=None, is_dense_gene_set=None, ps=None, sigma2s=None, assume_independent=False, num_missing_gene_sets=None, debug_genes=None, debug_gene_sets=None):

#     debug_gene_sets = None

#     if max_num_burn_in is None:
#         max_num_burn_in = int(max_num_iter * .25)
#     if max_num_burn_in >= max_num_iter:
#         max_num_burn_in = int(max_num_iter * .25)

#     #if (update_hyper_p or update_hyper_sigma) and gauss_seidel:
#     #    log("Using Gibbs sampling for betas since update hyper was requested")
#     #    gauss_seidel = False

#     if ses is None:
#         ses = self.ses
#     if beta_tildes is None:
#         beta_tildes = self.beta_tildes
        
#     if X_orig is None and not assume_independent:
#         X_orig = self.X_orig
#     if scale_factors is None:
#         scale_factors = self.scale_factors
#     if mean_shifts is None:
#         mean_shifts = self.mean_shifts

#     use_X = False
#     if V is None and not assume_independent:
#         if X_orig is None or scale_factors is None or mean_shifts is None:
#             bail("Require X, scale, and mean if V is None")
#         else:
#             use_X = True
#             log("Using low memory X instead of V", TRACE)

#     if is_dense_gene_set is None:
#         is_dense_gene_set = self.is_dense_gene_set
#     if ps is None:
#         ps = self.ps
#     if sigma2s is None:
#         sigma2s = self.sigma2s

#     if self.sigma2 is None:
#         bail("Need sigma to calculate betas!")

#     if initial_p is not None:
#         self.set_p(initial_p)

#     if self.p is None and ps is None:
#         bail("Need p to calculate non-inf betas")

#     if not len(beta_tildes.shape) == len(ses.shape):
#         bail("If running parallel beta inference, beta_tildes and ses must have same shape")

#     if len(beta_tildes.shape) == 0 or beta_tildes.shape[0] == 0:
#         bail("No gene sets are left!")

#     #convert the beta_tildes and ses to matrices -- columns are num_parallel
#     #they are always stored as matrices, with 1 column as needed
#     #V on the other hand will be a 2-D matrix if it is constant across all parallel (or if there is only 1)
#     #checking len(V.shape) can therefore distinguish a constant from variable V

#     multiple_V = False
#     sparse_V = False

#     if len(beta_tildes.shape) > 1:
#         num_gene_sets = beta_tildes.shape[1]

#         if not beta_tildes.shape[0] == ses.shape[0]:
#             bail("beta_tildes and ses must have same number of parallel runs")

#         #dimensions should be num_gene_sets, num_parallel
#         num_parallel = beta_tildes.shape[0]
#         beta_tildes_m = copy.copy(beta_tildes)
#         ses_m = copy.copy(ses)

#         if V is not None and type(V) is sparse.csc_matrix:
#             sparse_V = True
#             multiple_V = False
#         elif V is not None and len(V.shape) == 3:
#             if not V.shape[0] == beta_tildes.shape[0]:
#                 bail("V must have same number of parallel runs as beta_tildes")
#             multiple_V = True
#             sparse_V = False
#         else:
#             multiple_V = False
#             sparse_V = False

#     else:
#         num_gene_sets = len(beta_tildes)
#         if V is not None and type(V) is sparse.csc_matrix:
#             num_parallel = 1
#             multiple_V = False
#             sparse_V = True
#             beta_tildes_m = beta_tildes[np.newaxis,:]
#             ses_m = ses[np.newaxis,:]
#         elif V is not None and len(V.shape) == 3:
#             num_parallel = V.shape[0]
#             multiple_V = True
#             sparse_V = False
#             beta_tildes_m = np.tile(beta_tildes, num_parallel).reshape((num_parallel, len(beta_tildes)))
#             ses_m = np.tile(ses, num_parallel).reshape((num_parallel, len(ses)))
#         else:
#             num_parallel = 1
#             multiple_V = False
#             sparse_V = False
#             beta_tildes_m = beta_tildes[np.newaxis,:]
#             ses_m = ses[np.newaxis,:]

#     if num_parallel == 1 and multiple_V:
#         multiple_V = False
#         V = V[0,:,:]

#     if multiple_V:
#         assert(not use_X)

#     if scale_factors.shape != mean_shifts.shape:
#         bail("scale_factors must have same dimension as mean_shifts")

#     if len(scale_factors.shape) == 2 and not scale_factors.shape[0] == num_parallel:
#         bail("scale_factors must have same number of parallel runs as beta_tildes")
#     elif len(scale_factors.shape) == 1 and num_parallel == 1:
#         scale_factors_m = scale_factors[np.newaxis,:]
#         mean_shifts_m = mean_shifts[np.newaxis,:]
#     elif len(scale_factors.shape) == 1 and num_parallel > 1:
#         scale_factors_m = np.tile(scale_factors, num_parallel).reshape((num_parallel, len(scale_factors)))
#         mean_shifts_m = np.tile(mean_shifts, num_parallel).reshape((num_parallel, len(mean_shifts)))
#     else:
#         scale_factors_m = copy.copy(scale_factors)
#         mean_shifts_m = copy.copy(mean_shifts)

#     if len(is_dense_gene_set.shape) == 2 and not is_dense_gene_set.shape[0] == num_parallel:
#         bail("is_dense_gene_set must have same number of parallel runs as beta_tildes")
#     elif len(is_dense_gene_set.shape) == 1 and num_parallel == 1:
#         is_dense_gene_set_m = is_dense_gene_set[np.newaxis,:]
#     elif len(is_dense_gene_set.shape) == 1 and num_parallel > 1:
#         is_dense_gene_set_m = np.tile(is_dense_gene_set, num_parallel).reshape((num_parallel, len(is_dense_gene_set)))
#     else:
#         is_dense_gene_set_m = copy.copy(is_dense_gene_set)

#     if ps is not None:
#         if len(ps.shape) == 2 and not ps.shape[0] == num_parallel:
#             bail("ps must have same number of parallel runs as beta_tildes")
#         elif len(ps.shape) == 1 and num_parallel == 1:
#             ps_m = ps[np.newaxis,:]
#         elif len(ps.shape) == 1 and num_parallel > 1:
#             ps_m = np.tile(ps, num_parallel).reshape((num_parallel, len(ps)))
#         else:
#             ps_m = copy.copy(ps)
#     else:
#         ps_m = self.p

#     if sigma2s is not None:
#         if len(sigma2s.shape) == 2 and not sigma2s.shape[0] == num_parallel:
#             bail("sigma2s must have same number of parallel runs as beta_tildes")
#         elif len(sigma2s.shape) == 1 and num_parallel == 1:
#             orig_sigma2_m = sigma2s[np.newaxis,:]
#         elif len(sigma2s.shape) == 1 and num_parallel > 1:
#             orig_sigma2_m = np.tile(sigma2s, num_parallel).reshape((num_parallel, len(sigma2s)))
#         else:
#             orig_sigma2_m = copy.copy(sigma2s)
#     else:
#         orig_sigma2_m = self.sigma2

#     #for efficiency, batch genes to be updated each cycle
#     if assume_independent:
#         gene_set_masks = [np.full(beta_tildes_m.shape[1], True)]
#     else:
#         gene_set_masks = self._compute_gene_set_batches(V, X_orig=X_orig, mean_shifts=mean_shifts, scale_factors=scale_factors, use_sum=True, max_allowed_batch_correlation=max_allowed_batch_correlation)
        
#     sizes = [float(np.sum(x)) / (num_parallel if multiple_V else 1) for x in gene_set_masks]
#     log("Analyzing %d gene sets in %d batches of gene sets; size range %d - %d" % (num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes)  if len(sizes) > 0 else 0), DEBUG)

#     #get the dimensions of the gene_set_masks to match those of the betas
#     if num_parallel == 1:
#         assert(not multiple_V)
#         #convert the vectors into matrices with one dimension
#         gene_set_masks = [x[np.newaxis,:] for x in gene_set_masks]
#     elif not multiple_V:
#         #we have multiple parallel but only one V
#         gene_set_masks = [np.tile(x, num_parallel).reshape((num_parallel, len(x))) for x in gene_set_masks]

#     #variables are denoted
#     #v: vectors of dimension equal to the number of gene sets
#     #m: data that varies by parallel runs and gene sets
#     #t: data that varies by chains, parallel runs, and gene sets

#     #rules:
#     #1. adding a lower dimensional tensor to higher dimenional ones means final dimensions must match. These operations are usually across replicates
#     #2. lower dimensional masks on the other hand index from the beginning dimensions (can use :,:,mask to index from end)
    
#     tensor_shape = (num_chains, num_parallel, num_gene_sets)
#     matrix_shape = (num_parallel, num_gene_sets)

#     #these are current posterior means (including p and the conditional beta). They are used to calculate avg_betas
#     #using these as the actual betas would yield the Gauss-seidel algorithm
#     curr_post_means_t = np.zeros(tensor_shape)
#     curr_postp_t = np.ones(tensor_shape)

#     #these are the current betas to be used in each iteration
#     initial_sd = np.std(beta_tildes_m)
#     if initial_sd == 0:
#         initial_sd = 1

#     curr_betas_t = scipy.stats.norm.rvs(0, initial_sd, tensor_shape)

#     res_beta_hat_t = np.zeros(tensor_shape)

#     avg_betas_m = np.zeros(matrix_shape)
#     avg_betas2_m = np.zeros(matrix_shape)
#     avg_postp_m = np.zeros(matrix_shape)
#     num_avg = 0

#     #these are the posterior betas averaged across iterations
#     sum_betas_t = np.zeros(tensor_shape)
#     sum_betas2_t = np.zeros(tensor_shape)

#     # Setting up constants
#     #hyperparameters
#     #shrinkage prior
#     if self.sigma_power is not None:
#         #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
#         sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

#         #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
#         if np.sum(is_dense_gene_set_m) > 0:
#             if np.sum(~is_dense_gene_set_m) > 0:
#                 #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
#                 sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
#             else:
#                 #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
#                 sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), self.sigma2, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

#     else:
#         sigma2_m = orig_sigma2_m

#     if ps_m is not None and np.min(ps_m) != np.max(ps_m):
#         p_text = "mean p=%.3g (%.3g-%.3g)" % (self.p, np.min(ps_m), np.max(ps_m))
#     else:
#         p_text = "p=%.3g" % (self.p)
#     if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
#         sigma2_text = "mean sigma=%.3g (%.3g-%.3g)" % (self.sigma2, np.min(orig_sigma2_m), np.max(orig_sigma2_m))
#     else:
#         sigma2_text = "sigma=%.3g" % (self.sigma2)

#     if np.min(orig_sigma2_m) != np.max(orig_sigma2_m):
#         sigma2_p_text = "mean sigma2/p=%.3g (%.3g-%.3g)" % (self.sigma2/self.p, np.min(orig_sigma2_m/ps_m), np.max(orig_sigma2_m/ps_m))
#     else:
#         sigma2_p_text = "sigma2/p=%.3g" % (self.sigma2/self.p)


#     tag = ""
#     if assume_independent:
#         tag = "independent "
#     elif sparse_V:
#         tag = "partially independent "
        
#     log("Calculating %snon-infinitesimal betas with %s, %s; %s" % (tag, p_text, sigma2_text, sigma2_p_text))

#     #generate the diagonals to use per replicate
#     if assume_independent:
#         V_diag_m = None
#         account_for_V_diag_m = False
#     else:
#         if V is not None:
#             if num_parallel > 1:
#                 #dimensions are num_parallel, num_gene_sets, num_gene_sets
#                 if multiple_V:
#                     V_diag_m = np.diagonal(V, axis1=1, axis2=2)
#                 else:
#                     if sparse_V:
#                         V_diag = V.diagonal()
#                     else:
#                         V_diag = np.diag(V)
#                     V_diag_m = np.tile(V_diag, num_parallel).reshape((num_parallel, len(V_diag)))
#             else:
#                 if sparse_V:
#                     V_diag_m = V.diagonal()[np.newaxis,:]                        
#                 else:
#                     V_diag_m = np.diag(V)[np.newaxis,:]

#             account_for_V_diag_m = not np.isclose(V_diag_m, np.ones(matrix_shape)).all()
#         else:
#             #we compute it from X, so we know it is always 1
#             V_diag_m = None
#             account_for_V_diag_m = False

#     se2s_m = np.power(ses_m,2)

#     #the below code is based off of the LD-pred code for SNP PRS
#     iteration_num = 0
#     burn_in_phase_v = np.array([True for i in range(num_parallel)])


#     if betas_trace_out is not None:
#         betas_trace_fh = open_gz(betas_trace_out, 'w')
#         betas_trace_fh.write("It\tParallel\tChain\tGene_Set\tbeta_post\tbeta\tpostp\tres_beta_hat\tbeta_tilde\tbeta_internal\tres_beta_hat_internal\tbeta_tilde_internal\tse_internal\tsigma2\tp\tR\tR_weighted\tSEM\n")

#     prev_betas_m = None
#     sigma_underflow = False
#     printed_warning_swing = False
#     printed_warning_increase = False
#     while iteration_num < max_num_iter:  #Big iteration

#         #if some have not converged, only sample for those that have not converged (for efficiency)
#         compute_mask_v = copy.copy(burn_in_phase_v)
#         if np.sum(compute_mask_v) == 0:
#             compute_mask_v[:] = True

#         hdmp_m = (sigma2_m / ps_m)
#         hdmpn_m = hdmp_m + se2s_m
#         hdmp_hdmpn_m = (hdmp_m / hdmpn_m)

#         norm_scale_m = np.sqrt(np.multiply(hdmp_hdmpn_m, se2s_m))
#         c_const_m = (ps_m / np.sqrt(hdmpn_m))

#         d_const_m = (1 - ps_m) / ses_m

#         iteration_num += 1

#         #default to 1
#         curr_postp_t[:,compute_mask_v,:] = np.ones(tensor_shape)[:,compute_mask_v,:]

#         #sample whether each gene set has non-zero effect
#         rand_ps_t = np.random.random(tensor_shape)
#         #generate normal random variable sampling
#         rand_norms_t = scipy.stats.norm.rvs(0, 1, tensor_shape)

#         for gene_set_mask_ind in range(len(gene_set_masks)):

#             #the challenge here is that gene_set_mask_m produces a ragged (non-square) tensor
#             #so we are going to "flatten" the last two dimensions
#             #this requires some care, in particular when running einsum, which requires a square tensor

#             gene_set_mask_m = gene_set_masks[gene_set_mask_ind]
            
#             if debug_gene_sets is not None:
#                 cur_debug_gene_sets = [debug_gene_sets[i] for i in range(len(debug_gene_sets)) if gene_set_mask_m[0,i]]

#             #intersect compute_max_v with the rows of gene_set_mask (which are the parallel runs)
#             compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

#             current_num_parallel = sum(compute_mask_v)

#             #Value to use when determining if we should force an alpha shrink if estimates are way off compared to heritability estimates.  (Improves MCMC convergence.)
#             #zero_jump_prob=0.05
#             #frac_betas_explained = max(0.00001,np.sum(np.apply_along_axis(np.mean, 0, np.power(curr_betas_m,2)))) / self.y_var
#             #frac_sigma_explained = self.sigma2_total_var / self.y_var
#             #alpha_shrink = min(1 - zero_jump_prob, 1.0 / frac_betas_explained, (frac_sigma_explained + np.mean(np.power(ses[i], 2))) / frac_betas_explained)
#             alpha_shrink = 1

#             #subtract out the predicted effects of the other betas
#             #we need to zero out diagonal of V to do this, but rather than do this we will add it back in

#             #1. First take the union of the current_gene_set_mask
#             #this is to allow us to run einsum
#             #we are going to do it across more gene sets than are needed, and then throw away the computations that are extra for each batch
#             compute_mask_union = np.any(compute_mask_m, axis=0)

#             #2. Retain how to filter from the union down to each mask
#             compute_mask_union_filter_m = compute_mask_m[:,compute_mask_union]

#             if assume_independent:
#                 res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
#             else:
#                 if multiple_V:

#                     #3. Do einsum across the union
#                     #This does pointwise matrix multiplication of curr_betas_t (sliced on axis 1) with V (sliced on axis 0), maintaining axis 0 for curr_betas_t
#                     res_beta_hat_union_t = np.einsum('hij,ijk->hik', curr_betas_t[:,compute_mask_v,:], V[compute_mask_v,:,:][:,:,compute_mask_union]).reshape((num_chains, current_num_parallel, np.sum(compute_mask_union)))

#                 elif sparse_V:
#                     res_beta_hat_union_t = V[compute_mask_union,:].dot(curr_betas_t[:,compute_mask_v,:].T.reshape((curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0]))).reshape((np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
#                 elif use_X:
#                     if len(compute_mask_union.shape) == 2:
#                         assert(compute_mask_union.shape[0] == 1)
#                         compute_mask_union = np.squeeze(compute_mask_union)
#                     #curr_betas_t: (num_chains, num_parallel, num_gene_sets)
#                     #X_orig: (num_genes, num_gene_sets)
#                     #X_orig_t: (num_gene_sets, num_genes)
#                     #mean_shifts_m: (num_parallel, num_gene_sets)
#                     #curr_betas_filtered_t: (num_chains, num_compute, num_gene_sets)

#                     curr_betas_filtered_t = curr_betas_t[:,compute_mask_v,:] / scale_factors_m[compute_mask_v,:]

#                     #have to reshape latter two dimensions before multiplying because sparse matrix can only handle 2-D

#                     #interm = np.zeros((X_orig.shape[0],np.sum(compute_mask_v),curr_betas_t.shape[0]))
#                     #interm[:,compute_mask_v,:] = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

#                     interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],curr_betas_filtered_t.shape[0] * curr_betas_filtered_t.shape[1]))).reshape((X_orig.shape[0],curr_betas_filtered_t.shape[1],curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v,:] * curr_betas_filtered_t, axis=2).T

#                     #interm: (num_genes, num_parallel remaining, num_chains)

#                     #num_gene sets, num_parallel, num_chains

#                     #this broke under some circumstances when a parallel chain converged before the others
#                     res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,compute_mask_v,np.newaxis] * np.sum(interm, axis=0)).T
#                     res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[compute_mask_v,:][:,compute_mask_union])

#                     #res_beta_hat_union_t = (X_orig[:,compute_mask_union].T.dot(interm.reshape((interm.shape[0],interm.shape[1]*interm.shape[2]))).reshape((np.sum(compute_mask_union),interm.shape[1],interm.shape[2])) - mean_shifts_m.T[compute_mask_union,:][:,:,np.newaxis] * np.sum(interm, axis=0)).T
#                     #res_beta_hat_union_t /= (X_orig.shape[0] * scale_factors_m[:,compute_mask_union])

#                 else:
#                     res_beta_hat_union_t = curr_betas_t[:,compute_mask_v,:].dot(V[:,compute_mask_union])

#                 if betas_trace_out is not None and betas_trace_gene_sets is not None:
#                     all_map = self._construct_map_to_ind(betas_trace_gene_sets)
#                     cur_sets = [betas_trace_gene_sets[x] for x in range(len(betas_trace_gene_sets)) if compute_mask_union[x]]
#                     cur_map = self._construct_map_to_ind(cur_sets)

#                 #4. Now restrict to only the actual masks (which flattens things because the compute_mask_m is not square)

#                 res_beta_hat_t_flat = res_beta_hat_union_t[:,compute_mask_union_filter_m[compute_mask_v,:]]
#                 assert(res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m))

#                 #dimensions of res_beta_hat_t_flat are (num_chains, np.sum(compute_mask_m))
#                 #dimensions of beta_tildes_m are (num_parallel, num_gene_sets))
#                 #subtraction will subtract matrix from each of the matrices in the tensor

#                 res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

#                 if account_for_V_diag_m:
#                     #dimensions of V_diag_m are (num_parallel, num_gene_sets)
#                     #curr_betas_t is (num_chains, num_parallel, num_gene_sets)
#                     res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:,compute_mask_m]
#                 else:
#                     res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:,compute_mask_m]
            
#             b2_t_flat = np.power(res_beta_hat_t_flat, 2)
#             d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
#             numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
#             numerator_zero_mask_t_flat = (numerator_t_flat == 0)
#             denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
#             denominator_t_flat[numerator_zero_mask_t_flat] = 1


#             d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
#             numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

#             if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):

#                 warn("Detected imaginary numbers!")
#                 #if d is imaginary, we set it to 1
#                 denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
#                 #if d is real and numerator is imaginary, we set to 0 (both numerator and denominator will be imaginary)
#                 numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

#                 #Original code for handling edge cases; adapted above
#                 #Commenting these out for now, but they are here in case we ever detect non real numbers
#                 #if need them, masked_array is too inefficient -- change to real mask
#                 #d_real_mask_t = np.isreal(d_const_b2_exp_t)
#                 #numerator_real_mask_t = np.isreal(numerator_t)
#                 #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_not(d_real_mask_t), fill_value = 1).filled()
#                 #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(d_real_mask_t, np.logical_not(numerator_real_mask_t)), fill_value=0).filled()
#                 #curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(np.logical_and(d_real_mask_t, numerator_real_mask_t), numerator_zero_mask_t), fill_value=0).filled()



#             curr_postp_t[:,compute_mask_m] = (numerator_t_flat / denominator_t_flat)


#             #calculate current posterior means
#             #the left hand side, because it is masked, flattens the latter two dimensions into one
#             #so we flatten the result of the right hand size to a 1-D array to match up for the assignment
#             curr_post_means_t[:,compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (curr_postp_t[:,compute_mask_m] * res_beta_hat_t_flat)

                
#             if gauss_seidel:
#                 proposed_beta_t_flat = curr_post_means_t[:,compute_mask_m]
#             else:
#                 norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat

#                 #draw from the conditional distribution
#                 proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:,compute_mask_m]

#                 #set things to zero that sampled below p
#                 zero_mask_t_flat = rand_ps_t[:,compute_mask_m] >= curr_postp_t[:,compute_mask_m] * alpha_shrink
#                 proposed_beta_t_flat[zero_mask_t_flat] = 0

#             #update betas
#             #do this inside loop since this determines the res_beta
#             #same idea as above for collapsing
#             curr_betas_t[:,compute_mask_m] = proposed_beta_t_flat
#             res_beta_hat_t[:,compute_mask_m] = res_beta_hat_t_flat

#             #if debug_gene_sets is not None:
#             #    my_cur_tensor_shape = (1 if assume_independent else num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
#             #    my_cur_tensor_shape2 = (num_chains, current_num_parallel, np.sum(gene_set_mask_m[0,]))
#             #    my_res_beta_hat_t = res_beta_hat_t_flat.reshape(my_cur_tensor_shape)
#             #    my_proposed_beta_t = proposed_beta_t_flat.reshape(my_cur_tensor_shape2)
#             #    my_norm_mean_t = norm_mean_t_flat.reshape(my_cur_tensor_shape)
#             #    top_set = [cur_debug_gene_sets[i] for i in range(len(cur_debug_gene_sets)) if np.abs(my_res_beta_hat_t[0,0,i]) == np.max(np.abs(my_res_beta_hat_t[0,0,:]))][0]
#             #    log("TOP IS",top_set)
#             #    gs = set([ "mp_absent_T_cells", top_set])
#             #    ind = [i for i in range(len(cur_debug_gene_sets)) if cur_debug_gene_sets[i] in gs]
#             #    for i in ind:
#             #        log("BETA_TILDE",cur_debug_gene_sets[i],beta_tildes_m[0,i]/scale_factors_m[0,i])
#             #        log("Z",cur_debug_gene_sets[i],beta_tildes_m[0,i]/ses_m[0,i])
#             #        log("RES",cur_debug_gene_sets[i],my_res_beta_hat_t[0,0,i]/scale_factors_m[0,i])
#             #        #log("RESF",cur_debug_gene_sets[i],res_beta_hat_t_flat[i]/scale_factors_m[0,i])
#             #        log("NORM_MEAN",cur_debug_gene_sets[i],my_norm_mean_t[0,0,i])
#             #        log("NORM_SCALE_M",cur_debug_gene_sets[i],norm_scale_m[0,i])
#             #        log("RAND_NORMS",cur_debug_gene_sets[i],rand_norms_t[0,0,i])
#             #        log("PROP",cur_debug_gene_sets[i],my_proposed_beta_t[0,0,i]/scale_factors_m[0,i])
#             #        ind2 = [j for j in range(len(debug_gene_sets)) if debug_gene_sets[j] == cur_debug_gene_sets[i]]
#             #        for j in ind2:
#             #            log("POST",cur_debug_gene_sets[i],curr_post_means_t[0,0,j]/scale_factors_m[0,i])
#             #            log("SIGMA",sigma2_m if type(sigma2_m) is float or type(sigma2_m) is np.float64 else sigma2_m[0,i])
#             #            log("P",cur_debug_gene_sets[i],curr_postp_t[0,0,j],self.p)
#             #            log("HDMP",hdmp_m/np.square(scale_factors_m[0,i]) if type(hdmp_m) is float or type(hdmp_m) is np.float64 else hdmp_m[0,0]/np.square(scale_factors_m[0,i]))
#             #            log("SES",se2s_m[0,0]/np.square(scale_factors_m[0,i]))
#             #            log("HDMPN",hdmpn_m/np.square(scale_factors_m[0,i]) if type(hdmpn_m) is float or type(hdmpn_m) is np.float64 else hdmpn_m[0,0]/scale_factors_m[0,i])
#             #            log("HDMP_HDMPN",hdmp_hdmpn_m if type(hdmp_hdmpn_m) is float or type(hdmp_hdmpn_m) is np.float64 else hdmp_hdmpn_m[0,0])
#             #            log("NOW1",debug_gene_sets[j],curr_betas_t[0,0,j]/scale_factors_m[0,i])


#         if sparse_solution:
#             sparse_mask_t = curr_postp_t < ps_m

#             if sparse_frac_betas is not None:
#                 #zero out very small values relative to top or median
#                 relative_value = np.max(np.abs(curr_post_means_t), axis=2)
#                 sparse_mask_t = np.logical_or(sparse_mask_t, (np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T)

#             #don't set anything not currently computed
#             sparse_mask_t[:,np.logical_not(compute_mask_v),:] = False
#             log("Setting %d entries to zero due to sparsity" % (np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0))), TRACE)
#             curr_betas_t[sparse_mask_t] = 0
#             curr_post_means_t[sparse_mask_t] = 0

#             if debug_gene_sets is not None:
#                 ind = [i for i in range(len(debug_gene_sets)) if debug_gene_sets[i] in gs]

#         curr_betas_m = np.mean(curr_post_means_t, axis=0)
#         curr_postp_m = np.mean(curr_postp_t, axis=0)
#         #no state should be preserved across runs, but take a random one just in case
#         sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]),:,:]
#         sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]),:,:]
#         sum_betas_t[:,compute_mask_v,:] = sum_betas_t[:,compute_mask_v,:] + curr_post_means_t[:,compute_mask_v,:]
#         sum_betas2_t[:,compute_mask_v,:] = sum_betas2_t[:,compute_mask_v,:] + np.square(curr_post_means_t[:,compute_mask_v,:])

#         #now calculate the convergence metrics
#         R_m = np.zeros(matrix_shape)
#         beta_weights_m = np.zeros(matrix_shape)
#         sem2_m = np.zeros(matrix_shape)
#         will_break = False
#         if assume_independent:
#             burn_in_phase_v[:] = False
#         elif gauss_seidel:
#             if prev_betas_m is not None:
#                 sum_diff = np.sum(np.abs(prev_betas_m - curr_betas_m))
#                 sum_prev = np.sum(np.abs(prev_betas_m))
#                 tot_diff = sum_diff / sum_prev
#                 log("Iteration %d: gauss seidel difference = %.4g / %.4g = %.4g" % (iteration_num+1, sum_diff, sum_prev, tot_diff), TRACE)
#                 if iteration_num > min_num_iter and tot_diff < eps:
#                     burn_in_phase_v[:] = False
#                     log("Converged after %d iterations" % (iteration_num+1), INFO)
#             prev_betas_m = curr_betas_m
#         elif iteration_num > min_num_iter and np.sum(burn_in_phase_v) > 0:
#             def __calculate_R_tensor(sum_t, sum2_t, num):

#                 #mean of betas across all iterations; psi_dot_j
#                 mean_t = sum_t / float(num)

#                 #mean of betas across replicates; psi_dot_dot
#                 mean_m = np.mean(mean_t, axis=0)
#                 #variances of betas across all iterators; s_j
#                 var_t = (sum2_t - float(num) * np.power(mean_t, 2)) / (float(num) - 1)
#                 #B_v = (float(iteration_num) / (num_chains - 1)) * np.apply_along_axis(np.sum, 0, np.apply_along_axis(lambda x: np.power(x - mean_betas_v, 2), 1, mean_betas_m))
#                 B_m = (float(num) / (mean_t.shape[0] - 1)) * np.sum(np.power(mean_t - mean_m, 2), axis=0)
#                 W_m = (1.0 / float(mean_t.shape[0])) * np.sum(var_t, axis=0)
#                 avg_W_m = (1.0 / float(mean_t.shape[2])) * np.sum(var_t, axis=2)
#                 var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
#                 var_given_y_m[var_given_y_m < 0] = 0

#                 R_m = np.ones(W_m.shape)
#                 R_non_zero_mask_m = W_m > 0

#                 var_given_y_m[var_given_y_m < 0] = 0

#                 R_m[R_non_zero_mask_m] = np.sqrt(var_given_y_m[R_non_zero_mask_m] / W_m[R_non_zero_mask_m])
                
#                 return (B_m, W_m, R_m, avg_W_m, mean_t)

#             #these matrices have convergence statistics in format (num_parallel, num_gene_sets)
#             #WARNING: only the results for compute_mask_v are valid
#             (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num)

#             beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
#             sum_betas_t_mean = np.mean(sum_betas_t)
#             if sum_betas_t_mean > 0:
#                 np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

#             #calculate the thresholded / scaled R_v
#             num_R_above_1_v = np.sum(R_m >= 1, axis=1)
#             num_R_above_1_v[num_R_above_1_v == 0] = 1

#             #mean for each parallel run

#             R_m_above_1 = copy.copy(R_m)
#             R_m_above_1[R_m_above_1 < 1] = 0
#             mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

#             #max for each parallel run
#             max_index_v = np.argmax(R_m, axis=1)
#             max_index_parallel = None
#             max_val = None
#             for i in range(len(max_index_v)):
#                 if compute_mask_v[i] and (max_val is None or R_m[i,max_index_v[i]] > max_val):
#                     max_val = R_m[i,max_index_v[i]]
#                     max_index_parallel = i
#             max_R_v = np.max(R_m, axis=1)
            

#             #TEMP TEMP TEMP
#             #if priors_for_convergence:
#             #    curr_v = curr_betas_v
#             #    s_cur2_v = np.array([curr_v[i] for i in sorted(range(len(curr_v)), key=lambda k: -np.abs(curr_v[k]))])
#             #    s_cur2_v = np.square(s_cur2_v - np.mean(s_cur2_v))
#             #    cum_cur2_v = np.cumsum(s_cur2_v) / np.sum(s_cur2_v)
#             #    top_mask2 = np.array(cum_cur2_v < 0.99)
#             #    (B_v2, W_v2, R_v2) = __calculate_R(sum_betas_m[:,top_mask2], sum_betas2_m[:,top_mask2], iteration_num)
#             #    max_index2 = np.argmax(R_v2)
#             #    log("Iteration %d (betas): max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g" % (iteration_num, max_index2, B_v2[max_index2], W_v2[max_index2], R_v2[max_index2], np.mean(R_v2), np.sum(R_v2 > r_threshold_burn_in)), TRACE)
#             #END TEMP TEMP TEMP
                
#             if use_max_r_for_convergence:
#                 convergence_statistic_v = max_R_v
#             else:
#                 convergence_statistic_v = mean_thresholded_R_v

#             outlier_mask_m = np.full(avg_W_m.shape, False)
#             if avg_W_m.shape[0] > 10:
#                 #check the variances
#                 q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
#                 iqr_mask = q3 > q1
#                 chain_iqr_m = np.zeros(avg_W_m.shape)
#                 chain_iqr_m[:,iqr_mask] = (avg_W_m[:,iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
#                 #dimensions chain x parallel
#                 outlier_mask_m = beta_outlier_iqr_threshold
#                 if np.sum(outlier_mask_m) > 0:
#                     log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m), DEBUG)

#             if np.sum(R_m > 1) > 10:
#                 #check the Rs
#                 q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
#                 if q3 > q1:
#                     #Z score per parallel, gene
#                     R_iqr_m = (R_m - median) / (q3 - q1)
#                     #dimensions of parallel x gene sets
#                     bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
#                     bad_gene_sets_v = np.any(bad_gene_sets_m,0)
#                     if np.sum(bad_gene_sets_m) > 0:
#                         #now find the bad chains
#                         bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

#                         #np.where bad gene sets[0] lists parallel
#                         #bad chains lists the bad chain corresponding to each parallel
#                         cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
#                         cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

#                         log("Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)" % (np.sum(cur_outlier_mask_m), np.sum(np.any(cur_outlier_mask_m, axis=0)), np.sum(bad_gene_sets_m), np.min(R_m[bad_gene_sets_m]), np.max(R_m[bad_gene_sets_m]), np.min(R_iqr_m[bad_gene_sets_m]), np.max(R_iqr_m[bad_gene_sets_m])), DEBUG)
#                         outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

#                         #log("Outlier parallel: %s" % (np.where(bad_gene_sets_m)[0]), DEBUG)
#                         #log("Outlier values: %s" % (R_m[bad_gene_sets_m]), DEBUG)
#                         #log("Outlier IQR: %s" % (R_iqr_m[bad_gene_sets_m]), DEBUG)
#                         #log("Outlier chains: %s" % (bad_chains), DEBUG)


#                         #log("Actually in mask: %s" % (str(np.where(outlier_mask_m))))

#             non_outliers_m = ~outlier_mask_m
#             if np.sum(outlier_mask_m) > 0:
#                 log("Detected %d total outlier chains" % np.sum(outlier_mask_m), DEBUG)
#                 #dimensions are num_chains x num_parallel
#                 for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
#                     #find a non-outlier chain and replace the three matrices in the right place
#                     if np.sum(outlier_mask_m[:,outlier_parallel]) > 0:
#                         if np.sum(non_outliers_m[:,outlier_parallel]) > 0:
#                             replacement_chains = np.random.choice(np.where(non_outliers_m[:,outlier_parallel])[0], size=np.sum(outlier_mask_m[:,outlier_parallel]))
#                             log("Replaced chains %s with chains %s in parallel %d" % (np.where(outlier_mask_m[:,outlier_parallel])[0], replacement_chains, outlier_parallel), DEBUG)

#                             for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t, sum_betas2_t]:
#                                 tensor[outlier_mask_m[:,outlier_parallel],outlier_parallel,:] = copy.copy(tensor[replacement_chains,outlier_parallel,:])

#                         else:
#                             log("Every chain was an outlier so doing nothing", TRACE)


#             log("Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (iteration_num, (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[max_index_parallel], B_m[max_index_parallel, max_index_v[max_index_parallel]], W_m[max_index_parallel, max_index_v[max_index_parallel]], R_m[max_index_parallel, max_index_v[max_index_parallel]], np.mean(mean_thresholded_R_v), np.sum(R_m > r_threshold_burn_in)), TRACE)

#             converged_v = convergence_statistic_v < r_threshold_burn_in
#             newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
#             if np.sum(newly_converged_v) > 0:
#                 if num_parallel == 1:
#                     log("Converged after %d iterations" % iteration_num, INFO)
#                 else:
#                     log("Parallel %s converged after %d iterations" % (",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num), INFO)
#                 burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

#         if np.sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

#             if return_sample:

#                 frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
#                 if frac_increase > 0.01:
#                     warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
#                     printed_warning_increase = True

#                 frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
#                 if frac_opposite > 0.01:
#                     warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
#                     printed_warning_swing = False

#                 if np.sum(burn_in_phase_v) > 0:
#                     burn_in_phase_v[:] = False
#                     log("Stopping burn in after %d iterations" % (iteration_num), INFO)


#                 #max_beta = None
#                 #if max_beta is not None:
#                 #    threshold_ravel = max_beta * scale_factors_m.ravel()
#                 #    if np.sum(sample_betas_m.ravel() > threshold_ravel) > 0:
#                 #        log("Capped %d sample betas" % np.sum(sample_betas_m.ravel() > threshold_ravel), DEBUG)
#                 #        sample_betas_mask = sample_betas_m.ravel() > threshold_ravel
#                 #        sample_betas_m.ravel()[sample_betas_mask] = threshold_ravel[sample_betas_mask]
#                 #    if np.sum(curr_betas_m.ravel() > threshold_ravel) > 0:
#                 #        log("Capped %d curr betas" % np.sum(curr_betas_m.ravel() > threshold_ravel), DEBUG)
#                 #        curr_betas_mask = curr_betas_m.ravel() > threshold_ravel
#                 #        curr_betas_m.ravel()[curr_betas_mask] = threshold_ravel[curr_betas_mask]

#                 return (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m)

#             #average over the posterior means instead of samples
#             #these differ from sum_betas_v because those include the burn in phase
#             avg_betas_m += np.sum(curr_post_means_t, axis=0)
#             avg_betas2_m += np.sum(np.power(curr_post_means_t, 2), axis=0)
#             avg_postp_m += np.sum(curr_postp_t, axis=0)
#             num_avg += curr_post_means_t.shape[0]

#             if iteration_num >= min_num_iter and num_avg > 1:
#                 if gauss_seidel:
#                     will_break = True
#                 else:

#                     #calculate these here for trace printing
#                     avg_m = avg_betas_m
#                     avg2_m = avg_betas2_m
#                     sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
#                     sem2_v = np.sum(sem2_m, axis=0)
#                     zero_sem2_v = sem2_v == 0
#                     sem2_v[zero_sem2_v] = 1
#                     total_z_v = np.sqrt(np.sum(avg2_m / num_avg, axis=0) / sem2_v)
#                     total_z_v[zero_sem2_v] = np.inf

#                     log("Iteration %d: sum2=%.4g; sum sem2=%.4g; z=%.3g" % (iteration_num, np.sum(avg2_m / num_avg), np.sum(sem2_m), np.min(total_z_v)), TRACE)

#                     min_z_sampling_var = 10
#                     if np.all(total_z_v > min_z_sampling_var):
#                         log("Desired precision achieved; stopping sampling")
#                         will_break=True

#                     #TODO: STILL FINALIZING HOW TO DO THIS
#                     #avg_m = avg_betas_m
#                     #avg2_m = avg_betas2_m

#                     #sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
#                     #zero_sem2_m = sem2_m == 0
#                     #sem2_m[zero_sem2_m] = 1

#                     #max_avg = np.max(np.abs(avg_m / num_avg))
#                     #min_avg = np.min(np.abs(avg_m / num_avg))
#                     #ref_val = max_avg - min_avg
#                     #if ref_val == 0:
#                     #    ref_val = np.sqrt(np.var(curr_post_means_t))
#                     #    if ref_val == 0:
#                     #        ref_val = 1

#                     #max_sem = np.max(np.sqrt(sem2_m))
#                     #max_percentage_error = max_sem / ref_val

#                     #log("Iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num, ref_val, max_sem, max_percentage_error))
#                     #if max_percentage_error < max_frac_sem:
#                     #    log("Desired precision achieved; stopping sampling")
#                     #    break
                    
#         else:
#             if update_hyper_p or update_hyper_sigma:
#                 h2 = 0
#                 for i in range(num_parallel):
#                     if use_X:
#                         h2 += curr_betas_m[i,:].dot(curr_betas_m[i,:])
#                     else:
#                         if multiple_V:
#                             cur_V = V[i,:,:]
#                         else:
#                             cur_V = V
#                         if sparse_V:
#                             h2 += V.dot(curr_betas_m[i,:].T).T.dot(curr_betas_m[i,:])
#                         else:
#                             h2 += curr_betas_m[i,:].dot(cur_V).dot(curr_betas_m[i,:])
#                 h2 /= num_parallel

#                 new_p = np.mean((np.sum(curr_betas_t > 0, axis=2) + num_p_pseudo) / float(curr_betas_t.shape[2] + num_p_pseudo))

#                 if self.sigma_power is not None:
#                     new_sigma2 = h2 / np.mean(np.sum(np.power(scale_factors_m, self.sigma_power), axis=1))
#                 else:
#                     new_sigma2 = h2 / num_gene_sets

#                 if num_missing_gene_sets:
#                     missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
#                     new_sigma2 *= missing_scale_factor
#                     new_p *= missing_scale_factor

#                 if p_noninf_inflate != 1:
#                     log("Inflating p by %.3g" % p_noninf_inflate, DEBUG)
#                     new_p *= p_noninf_inflate

#                 if abs(new_sigma2 - self.sigma2) / self.sigma2 < eps and abs(new_p - self.p) / self.p < eps:
#                     log("Sigma converged to %.4g; p converged to %.4g" % (self.sigma2, self.p), TRACE)
#                     update_hyper_sigma = False
#                     update_hyper_p = False
#                 else:
#                     if update_hyper_p:
#                         log("Updating p from %.4g to %.4g" % (self.p, new_p), TRACE)
#                         if not update_hyper_sigma and adjust_hyper_sigma_p:
#                             #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
#                             #if we are only updating p, and adjusting sigma, we will leave the conditional_sigma constant, which means scaling the sigma
#                             new_sigma2 = self.sigma2 / self.p * new_p
#                             log("Updating sigma from %.4g to %.4g to maintain constant sigma/p" % (self.sigma2, new_sigma2), TRACE)
#                             #we need to adjust the total sigma to keep the conditional sigma constant
#                             self.set_sigma(new_sigma2, self.sigma_power)
#                         self.set_p(new_p)
                            
#                     if update_hyper_sigma:
#                         if not sigma_underflow:
#                             log("Updating sigma from %.4g to %.4g ( sqrt(sigma2/p)=%.4g )" % (self.sigma2, new_sigma2, np.sqrt(new_sigma2 / self.p)), TRACE)

#                         lower_bound = 2e-3

#                         if sigma_underflow or new_sigma2 / self.p < lower_bound:
                            
#                             #first, try the heuristic of setting sigma2 so that strongest gene set has maximum possible p_bar

#                             max_e_beta2 = np.argmax(beta_tildes_m / ses_m)

#                             max_se2 = se2s_m.ravel()[max_e_beta2]
#                             max_beta_tilde = beta_tildes_m.ravel()[max_e_beta2]
#                             max_beta_tilde2 = np.square(max_beta_tilde)

#                             #OLD inference
#                             #make sigma/p easily cover the observation
#                             #new_sigma2 = (max_beta_tilde2 - max_se2) * self.p
#                             #make sigma a little bit smaller so that the top gene set is a little more of an outlier
#                             #new_sigma2 /= sigma_num_devs_to_top

#                             #NEW inference
#                             max_beta = np.sqrt(max_beta_tilde2 - max_se2)
#                             correct_sigma2 = self.p * np.square(max_beta / np.abs(scipy.stats.norm.ppf(1 / float(curr_betas_t.shape[2]) * self.p * 2)))
#                             new_sigma2 = correct_sigma2

#                             if new_sigma2 / self.p <= lower_bound:
#                                 new_sigma2_from_top = new_sigma2
#                                 new_sigma2 = lower_bound * self.p
#                                 log("Sigma underflow including with determination from top gene set (%.4g)! Setting sigma to lower bound (%.4g * %.4g = %.4g) and no updates" % (new_sigma2_from_top, lower_bound, self.p, new_sigma2), TRACE)
#                             else:
#                                 log("Sigma underflow! Setting sigma determined from top gene set (%.4g) and no updates" % new_sigma2, TRACE)

#                             if self.sigma_power is not None:

#                                 #gene set specific sigma is internal sigma2 multiplied by scale_factor ** power
#                                 #new_sigma2 is final sigma
#                                 #so store internal value as final divided by average power

#                                 #use power learned from mouse
#                                 #using average across gene sets makes it sensitive to distribution of gene sets
#                                 #need better solution for learning; since we are hardcoding from top gene set, just use mouse value
#                                 new_sigma2 = new_sigma2 / np.power(self.MEAN_MOUSE_SCALE, self.sigma_power)

#                                 #if np.sum([~is_dense_gene_set_m]) > 0:
#                                 #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
#                                 #else:
#                                 #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m.ravel()), self.sigma_power)

#                                 #if is_dense_gene_set_m.ravel()[max_e_beta2]:
#                                 #    new_sigma2 = new_sigma2 / np.power(np.mean(scale_factors_m[~is_dense_gene_set_m].ravel()), self.sigma_power)
#                                 #else:
#                                 #    new_sigma2 = new_sigma2 / np.power(scale_factors_m.ravel()[max_e_beta2], self.sigma_power)

#                             if not update_hyper_p and adjust_hyper_sigma_p:
#                                 #remember, sigma is the *total* variance term. It is equal to p * conditional_sigma.
#                                 #if we are only sigma p, and adjusting p, we will leave the conditional_sigma constant, which means scaling the p
#                                 new_p = self.p / self.sigma2 * new_sigma2
#                                 log("Updating p from %.4g to %.4g to maintain constant sigma/p" % (self.p, new_p), TRACE)
#                                 #we need to adjust the total sigma to keep the conditional sigma constant
#                                 self.set_p(new_p)

#                             self.set_sigma(new_sigma2, self.sigma_power)
#                             sigma_underflow = True

#                             #update_hyper_sigma = False
#                             #restarting sampling with sigma2 fixed to initial value due to underflow
#                             #update_hyper_p = False

#                             #reset loop state
#                             #iteration_num = 0
#                             #curr_post_means_t = np.zeros(tensor_shape)
#                             #curr_postp_t = np.ones(tensor_shape)
#                             #curr_betas_t = scipy.stats.norm.rvs(0, np.std(beta_tildes_m), tensor_shape)                            
#                             #avg_betas_m = np.zeros(matrix_shape)
#                             #avg_betas2_m = np.zeros(matrix_shape)
#                             #avg_postp_m = np.zeros(matrix_shape)
#                             #num_avg = 0
#                             #sum_betas_t = np.zeros(tensor_shape)
#                             #sum_betas2_t = np.zeros(tensor_shape)
#                         else:
#                             self.set_sigma(new_sigma2, self.sigma_power)

#                         #update the matrix forms of these variables
#                         orig_sigma2_m *= new_sigma2 / np.mean(orig_sigma2_m)
#                         if self.sigma_power is not None:
#                             #sigma2_m = orig_sigma2_m * np.power(scale_factors_m, self.sigma_power)
#                             sigma2_m = self.get_scaled_sigma2(scale_factors_m, orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)

#                             #for dense gene sets, scaling by size doesn't make sense. So use mean size across sparse gene sets
#                             if np.sum(is_dense_gene_set_m) > 0:
#                                 if np.sum(~is_dense_gene_set_m) > 0:
#                                     #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m[~is_dense_gene_set_m]), self.sigma_power)
#                                     sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m[~is_dense_gene_set_m]), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
#                                 else:
#                                     #sigma2_m[is_dense_gene_set_m] = self.sigma2 * np.power(np.mean(scale_factors_m), self.sigma_power)
#                                     sigma2_m[is_dense_gene_set_m] = self.get_scaled_sigma2(np.mean(scale_factors_m), orig_sigma2_m, self.sigma_power, self.sigma_threshold_k, self.sigma_threshold_xo)
#                         else:
#                             sigma2_m = orig_sigma2_m

#                         ps_m *= new_p / np.mean(ps_m)

#         if betas_trace_out is not None:
#             for parallel_num in range(num_parallel):
#                 for chain_num in range(num_chains):
#                     for i in range(num_gene_sets):
#                         gene_set = i
#                         if betas_trace_gene_sets is not None and len(betas_trace_gene_sets) == num_gene_sets:
#                             gene_set = betas_trace_gene_sets[i]

#                         betas_trace_fh.write("%d\t%d\t%d\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g\n" % (iteration_num, parallel_num+1, chain_num+1, gene_set, curr_post_means_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], curr_postp_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i] / scale_factors_m[parallel_num,i], beta_tildes_m[parallel_num,i] / scale_factors_m[parallel_num,i], curr_betas_t[chain_num,parallel_num,i], res_beta_hat_t[chain_num,parallel_num,i], beta_tildes_m[parallel_num,i], ses_m[parallel_num,i], sigma2_m[parallel_num,i] if len(np.shape(sigma2_m)) > 0 else sigma2_m, ps_m[parallel_num,i] if len(np.shape(ps_m)) > 0 else ps_m, R_m[parallel_num,i], R_m[parallel_num,i] * beta_weights_m[parallel_num,i], sem2_m[parallel_num, i]))

#             betas_trace_fh.flush()

#         if will_break:
#             break


#     if betas_trace_out is not None:
#         betas_trace_fh.close()

#         #log("%d\t%s" % (iteration_num, "\t".join(["%.3g\t%.3g" % (curr_betas_m[i,0], (np.mean(sum_betas_m, axis=0) / iteration_num)[i]) for i in range(curr_betas_m.shape[0])])), TRACE)

#     avg_betas_m /= num_avg
#     avg_postp_m /= num_avg

#     if num_parallel == 1:
#         avg_betas_m = avg_betas_m.flatten()
#         avg_postp_m = avg_postp_m.flatten()

#     #max_beta = None
#     #if max_beta is not None:
#     #    threshold_ravel = max_beta * scale_factors_m.ravel()
#     #    if np.sum(avg_betas_m.ravel() > threshold_ravel) > 0:
#     #        log("Capped %d sample betas" % np.sum(avg_betas_m.ravel() > threshold_ravel), DEBUG)
#     #        avg_betas_mask = avg_betas_m.ravel() > threshold_ravel
#     #        avg_betas_m.ravel()[avg_betas_mask] = threshold_ravel[avg_betas_mask]

#     frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
#     if frac_increase > 0.01:
#         warn("A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
#         printed_warning_increase = True

#     frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
#     if frac_opposite > 0.01:
#         warn("A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
#         printed_warning_swing = False

#     return (avg_betas_m, avg_postp_m)


# def calculate_non_inf_betas(self, p, max_num_burn_in=1000, max_num_iter=1100, min_num_iter=10, num_chains=10, r_threshold_burn_in=1.01, use_max_r_for_convergence=True, max_frac_sem=0.01, gauss_seidel=False, update_hyper_sigma=True, update_hyper_p=True, sparse_solution=False, pre_filter_batch_size=None, pre_filter_small_batch_size=500, sparse_frac_betas=None, betas_trace_out=None, **kwargs):

#     log("Calculating betas")
#     (avg_betas_uncorrected_v, avg_postp_uncorrected_v) = self._calculate_non_inf_betas(p, max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=False, update_hyper_p=False, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, assume_independent=True, V=None, **kwargs)

#     avg_betas_v = np.zeros(len(self.gene_sets))
#     avg_postp_v = np.zeros(len(self.gene_sets))

#     initial_run_mask = avg_betas_uncorrected_v != 0
#     run_mask = copy.copy(initial_run_mask)

#     if pre_filter_batch_size is not None and np.sum(initial_run_mask) > pre_filter_batch_size:
#         self._record_param("pre_filter_batch_size_orig", pre_filter_batch_size)

#         num_batches = self._get_num_X_blocks(self.X_orig[:,initial_run_mask], batch_size=pre_filter_small_batch_size)
#         if num_batches > 1:
#             #try to run with small batches to see if we can zero out more
#             gene_set_masks = self._compute_gene_set_batches(V=None, X_orig=self.X_orig[:,initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], scale_factors=self.scale_factors[initial_run_mask], find_correlated_instead=pre_filter_small_batch_size)
#             if len(gene_set_masks) > 0:
#                 if np.sum(gene_set_masks[-1]) == 1 and len(gene_set_masks) > 1:
#                     #merge singletons at the end into the one before
#                     gene_set_masks[-2][gene_set_masks[-1]] = True
#                     gene_set_masks = gene_set_masks[:-1]
#                 if np.sum(gene_set_masks[0]) > 1:
#                     V_data = []
#                     V_rows = []
#                     V_cols = []
#                     for gene_set_mask in gene_set_masks:
#                         V_block = self._calculate_V_internal(self.X_orig[:,initial_run_mask][:,gene_set_mask], self.y_corr_cholesky, self.mean_shifts[initial_run_mask][gene_set_mask], self.scale_factors[initial_run_mask][gene_set_mask])
#                         orig_indices = np.where(gene_set_mask)[0]
#                         V_rows += list(np.repeat(orig_indices, V_block.shape[0]))
#                         V_cols += list(np.tile(orig_indices, V_block.shape[0]))
#                         V_data += list(V_block.ravel())
                        
#                     V_sparse = sparse.csc_matrix((V_data, (V_rows, V_cols)), shape=(np.sum(initial_run_mask), np.sum(initial_run_mask)))

#                     log("Running %d blocks to check for zeros..." % len(gene_set_masks), DEBUG)
#                     (avg_betas_half_corrected_v, avg_postp_half_corrected_v) = self._calculate_non_inf_betas(p, V=V_sparse, X_orig=None, scale_factors=self.scale_factors[initial_run_mask], mean_shifts=self.mean_shifts[initial_run_mask], is_dense_gene_set=self.is_dense_gene_set[initial_run_mask], ps=self.ps[initial_run_mask], sigma2s=self.sigma2s[initial_run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, **kwargs)

#                     add_zero_mask = avg_betas_half_corrected_v == 0

#                     if np.any(add_zero_mask):
#                         #need to convert these to the original gene sets
#                         map_to_full = np.where(initial_run_mask)[0]
#                         #get rows and then columns in subsetted
#                         set_to_zero_full = np.where(add_zero_mask)
#                         #map columns in subsetted to original
#                         set_to_zero_full = map_to_full[set_to_zero_full]
#                         orig_zero = np.sum(run_mask)
#                         run_mask[set_to_zero_full] = False
#                         new_zero = np.sum(run_mask)
#                         log("Found %d additional zero gene sets" % (orig_zero - new_zero),DEBUG)

#     if np.sum(~run_mask) > 0:
#         log("Set additional %d gene sets to zero based on uncorrected betas" % np.sum(~run_mask))

#     if np.sum(run_mask) == 0 and self.p_values is not None:
#         run_mask[np.argmax(self.p_values)] = True

#     (avg_betas_v[run_mask], avg_postp_v[run_mask]) = self._calculate_non_inf_betas(p, beta_tildes=self.beta_tildes[run_mask], ses=self.ses[run_mask], X_orig=self.X_orig[:,run_mask], scale_factors=self.scale_factors[run_mask], mean_shifts=self.mean_shifts[run_mask], V=None, ps=self.ps[run_mask] if self.ps is not None else None, sigma2s=self.sigma2s[run_mask] if self.sigma2s is not None else None, is_dense_gene_set=self.is_dense_gene_set[run_mask], max_num_burn_in=max_num_burn_in, max_num_iter=max_num_iter, min_num_iter=min_num_iter, num_chains=num_chains, r_threshold_burn_in=r_threshold_burn_in, use_max_r_for_convergence=use_max_r_for_convergence, max_frac_sem=max_frac_sem, gauss_seidel=gauss_seidel, update_hyper_sigma=update_hyper_sigma, update_hyper_p=update_hyper_p, sparse_solution=sparse_solution, sparse_frac_betas=sparse_frac_betas, betas_trace_out=betas_trace_out, betas_trace_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], debug_gene_sets=[self.gene_sets[i] for i in range(len(self.gene_sets)) if run_mask[i]], **kwargs)

#     if len(avg_betas_v.shape) == 2:
#         avg_betas_v = np.mean(avg_betas_v, axis=0)
#         avg_postp_v = np.mean(avg_postp_v, axis=0)

#     self.betas = copy.copy(avg_betas_v)
#     self.betas_uncorrected = copy.copy(avg_betas_uncorrected_v)

#     self.non_inf_avg_postps = copy.copy(avg_postp_v)
#     self.non_inf_avg_cond_betas = copy.copy(avg_betas_v)
#     self.non_inf_avg_cond_betas[avg_postp_v > 0] /= avg_postp_v[avg_postp_v > 0]

#     if self.gene_sets_missing is not None:
#         self.betas_missing = np.zeros(len(self.gene_sets_missing))
#         self.betas_uncorrected_missing = np.zeros(len(self.gene_sets_missing))
#         self.non_inf_avg_postps_missing = np.zeros(len(self.gene_sets_missing))
#         self.non_inf_avg_cond_betas_missing = np.zeros(len(self.gene_sets_missing))







# # The sparse matrix X needs to be genes (rows) x gene sets (columns)
# # There is code read_X that does this, but it is probably overkill
# # 
# # You’ll want to read into the sparse matrix at boot up of the server, and then store some auxiliary structures that map gene names to rows and gene set names to columns
# # (b) The user request will then be a list of genes and values
# # You’ll load that into a y vector of length equal to X.shape[0] with genes in the same order
# # (c) You can then run compute_beta_tildes which takes the X and y vectors as inputs (plus some quantities that can be easily computed from X)
# # This will return some values that allow you to filter the columns of the X matrix (I can show you how tomorrow)

# def run_factor(X_orig, max_num_factors=15, alpha0=10, beta0=1, gene_set_filter_type=None, gene_set_filter_value=None, gene_filter_type=None, gene_filter_value=None, gene_set_multiply_type=None, gene_multiply_type=None, run_transpose=True, max_num_iterations=100, rel_tol=0.01, lmm_auth_key=None):

#     BETA_UNCORRECTED_OPTIONS = ["beta_uncorrected", "betas_uncorrected"]
#     BETA_OPTIONS = ["beta", "betas"]
#     P_OPTIONS = ["p_value", "p", "p_value"]
#     NONE_OPTIONS = ["none"]

#     # self._record_params({"max_num_factors": max_num_factors, "alpha0": alpha0, "gene_set_filter_type": gene_set_filter_type, "gene_set_filter_value": gene_set_filter_value, "gene_filter_type": gene_filter_type, "gene_filter_value": gene_filter_value, "gene_set_multiply_type": gene_set_multiply_type, "gene_multiply_type": gene_multiply_type, "run_transpose": run_transpose})

#     gene_set_mask = np.full(self.X_orig.shape[1], True)
#     if gene_set_filter_type is not None and gene_set_filter_value is not None:
#         if gene_set_filter_type.lower() in BETA_UNCORRECTED_OPTIONS:
#             if self.betas_uncorrected is None:
#                 raise RunFactorException("Can't run filtering in factor; betas_uncorrected was not loaded")
#             gene_set_mask = self.betas_uncorrected > gene_set_filter_value
#         elif gene_set_filter_type.lower() in BETA_OPTIONS:
#             if self.betas is None:
#                 raise RunFactorException("Can't run filtering in factor; betas was not loaded")
#             gene_set_mask = self.betas > gene_set_filter_value
#         elif gene_set_filter_type.lower() in P_OPTIONS:
#             if self.p_values is None:
#                 raise RunFactorException("Can't run filtering in factor; p was not loaded")
#             gene_set_mask = self.p_values < gene_set_filter_value
#         elif gene_set_filter_type.lower() in NONE_OPTIONS:
#             pass
#         else:
#             raise RunFactorException("Valid values for --gene-set-filter-type are beta_uncorrected|beta|p_value")

#     gene_set_vector = np.ones(self.X_orig.shape[1])
#     if gene_set_multiply_type is not None:
#         if gene_set_multiply_type.lower() in BETA_UNCORRECTED_OPTIONS:
#             if self.betas_uncorrected is None:
#                 raise RunFactorException("Can't run multiply in factor; betas_uncorrected was not loaded")
#             gene_set_vector = self.betas_uncorrected
#         elif gene_set_multiply_type.lower() in BETA_OPTIONS:
#             if self.betas is None:
#                 raise RunFactorException("Can't run multiply in factor; betas was not loaded")
#             gene_set_vector = self.betas
#         elif gene_set_multiply_type.lower() in P_OPTIONS:
#             if self.p_values is None:
#                 raise RunFactorException("Can't run multiply in factor; p was not loaded")
#             gene_set_vector = -np.log(self.p_values)
#         elif gene_set_multiply_type.lower() in NONE_OPTIONS:
#             pass
#         else:
#             bail("Valid values for --gene-set-multiply-type are beta_uncorrected|beta|p_value")


#     PRIOR_OPTIONS = ["prior", "priors"]
#     COMBINED_OPTIONS = ["combined", "combined_prior_Y", "combined_prior_Ys"]
#     Y_OPTIONS = ["y", "log_bf"]

#     gene_mask = np.full(self.X_orig.shape[0], True)
#     if gene_filter_type is not None and gene_filter_value is not None:
#         if gene_filter_type.lower() in PRIOR_OPTIONS:
#             if self.priors is None:
#                 raise RunFactorException("Can't run filtering in factor; priors were not loaded")
#             gene_mask = self.priors > gene_filter_value
#         elif gene_filter_type.lower() in COMBINED_OPTIONS:
#             if self.combined_prior_Ys is None:
#                 raise RunFactorException("Can't run filtering in factor; combined was not loaded")
#             gene_mask = self.combined_prior_Ys > gene_filter_value
#         elif gene_filter_type.lower() in Y_OPTIONS:
#             if self.Y is None:
#                 raise RunFactorException("Can't run filtering in factor; log_bf was not loaded")
#             gene_mask = self.Y > gene_filter_value
#         elif gene_filter_type.lower() in NONE_OPTIONS:
#             pass
#         else:
#             raise RunFactorException("Valid values for --gene-filter-type are prior|combined|log_bf")

#     gene_vector = np.ones(self.X_orig.shape[0])
#     if gene_multiply_type is not None:
#         if gene_multiply_type.lower() in PRIOR_OPTIONS:
#             if self.priors is None:
#                 raise RunFactorException("Can't run multiply in factor; priors were not loaded")
#             gene_vector = self.priors
#         elif gene_multiply_type.lower() in COMBINED_OPTIONS:
#             if self.combined_prior_Ys is None:
#                 raise RunFactorException("Can't run multiply in factor; combined was not loaded")
#             gene_vector = self.combined_prior_Ys
#         elif gene_multiply_type.lower() in Y_OPTIONS:
#             if self.Y is None:
#                 raise RunFactorException("Can't run multiply in factor; log_bf was not loaded")
#             gene_vector = self.Y
#         elif gene_multiply_type.lower() in NONE_OPTIONS:
#             pass
#         else:
#             raise RunFactorException("Valid values for --gene-multiply-type are prior|combined|log_bf")

#     #make sure everything is positive
#     gene_set_vector[gene_set_vector < 0] = 0
#     gene_vector[gene_vector < 0] = 0

#     logger.info("Running matrix factorization")

#     matrix = self.X_orig[:,gene_set_mask][gene_mask,:].multiply(gene_set_vector[gene_set_mask]).toarray().T * gene_vector[gene_mask]
#     logger.info("Matrix shape: (%s, %s)" % (matrix.shape))

#     if not run_transpose:
#         matrix = matrix.T


#     #this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
#     def _bayes_nmf_l2(V0, n_iter=10000, a0=10, tol=1e-7, K=15, K0=15, phi=1.0):

#         # Bayesian NMF with half-normal priors for W and H
#         # V0: input z-score matrix (variants x traits)
#         # n_iter: Number of iterations for parameter optimization
#         # a0: Hyper-parameter for inverse gamma prior on ARD relevance weights
#         # tol: Tolerance for convergence of fitting procedure
#         # K: Number of clusters to be initialized (algorithm may drive some to zero)
#         # K0: Used for setting b0 (lambda prior hyper-parameter) -- should be equal to K
#         # phi: Scaling parameter

#         eps = 1.e-50
#         delambda = 1.0
#         #active_nodes = np.sum(V0, axis=0) != 0
#         #V0 = V0[:,active_nodes]
#         V = V0 - np.min(V0)
#         Vmin = np.min(V)
#         Vmax = np.max(V)
#         N = V.shape[0]
#         M = V.shape[1]

#         W = np.random.random((N, K)) * Vmax #NxK
#         H = np.random.random((K, M)) * Vmax #KxM

#         I = np.ones((N, M)) #NxM
#         V_ap = W.dot(H) + eps #NxM

#         phi = np.power(np.std(V), 2) * phi
#         C = (N + M) / 2 + a0 + 1
#         b0 = 3.14 * (a0 - 1) * np.mean(V) / (2 * K0)
#         lambda_bound = b0 / C
#         lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
#         lambda_cut = lambda_bound * 1.5

#         n_like = [None]
#         n_evid = [None]
#         n_error = [None]
#         n_lambda = [lambdak]
#         it = 1
#         count = 1
#         while delambda >= tol and it < n_iter:
#             H = H * (W.T.dot(V)) / (W.T.dot(V_ap) + phi * H * np.repeat(1/lambdak, M).reshape(len(lambdak), M) + eps)
#             V_ap = W.dot(H) + eps
#             W = W * (V.dot(H.T)) / (V_ap.dot(H.T) + phi * W * np.tile(1/lambdak, N).reshape(N, len(lambdak)) + eps)
#             V_ap = W.dot(H) + eps
#             lambdak = (0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / C
#             delambda = np.max(np.abs(lambdak - n_lambda[it - 1]) / n_lambda[it - 1])
#             like = np.sum(np.power(V - V_ap, 2)) / 2
#             n_like.append(like)
#             n_evid.append(like + phi * np.sum((0.5 * np.sum(np.power(W, 2), axis=0) + 0.5 * np.sum(np.power(H, 2), axis=1) + b0) / lambdak + C * np.log(lambdak)))
#             n_lambda.append(lambdak)
#             n_error.append(np.sum(np.power(V - V_ap, 2)))
#             if it % 100 == 0:
#                 log("Iteration=%d; evid=%.3g; lik=%.3g; err=%.3g; delambda=%.3g; factors=%d; factors_non_zero=%d" % (it, n_evid[it], n_like[it], n_error[it], delambda, np.sum(np.sum(W, axis=0) != 0), np.sum(lambdak >= lambda_cut)), TRACE)
#             it += 1

#         return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]
#             #W # Variant weight matrix (N x K)
#             #H # Trait weight matrix (K x M)
#             #n_like # List of reconstruction errors (sum of squared errors / 2) per iteration
#             #n_evid # List of negative log-likelihoods per iteration
#             #n_lambda # List of lambda vectors (shared weights for each of K clusters, some ~0) per iteration
#             #n_error # List of reconstruction errors (sum of squared errors) per iteration


#     result = _bayes_nmf_l2(matrix, a0=alpha0, K=max_num_factors, K0=max_num_factors)
    
#     self.exp_lambdak = result[4]
#     self.exp_gene_factors = result[1].T
#     self.exp_gene_set_factors = result[0]


#     #subset_down
#     factor_mask = self.exp_lambdak != 0 & (np.sum(self.exp_gene_factors, axis=0) > 0) & (np.sum(self.exp_gene_set_factors, axis=0) > 0)
#     factor_mask = factor_mask & (np.max(self.exp_gene_set_factors, axis=0) > 1e-5 * np.max(self.exp_gene_set_factors))

#     if np.sum(~factor_mask) > 0:
#         self.exp_lambdak = self.exp_lambdak[factor_mask]
#         self.exp_gene_factors = self.exp_gene_factors[:,factor_mask]
#         self.exp_gene_set_factors = self.exp_gene_set_factors[:,factor_mask]

#     self.gene_factor_gene_mask = gene_mask
#     self.gene_set_factor_gene_set_mask = gene_set_mask

#     gene_set_values = None
#     if self.betas is not None:
#         gene_set_values = self.betas
#     elif self.betas_uncorrected is not None:
#         gene_set_values = self.betas_uncorrected

#     gene_values = None
#     if self.combined_prior_Ys is not None:
#         gene_values = self.combined_prior_Ys
#     elif self.priors is not None:
#         gene_values = self.priors
#     elif self.Y is not None:
#         gene_values = self.Y

#     if gene_set_values is not None:
#         self.factor_gene_set_scores = self.exp_gene_set_factors.T.dot(gene_set_values[self.gene_set_factor_gene_set_mask])
#     else:
#         self.factor_gene_set_scores = self.exp_lambdak

#     if gene_values is not None:
#         self.factor_gene_scores = self.exp_gene_factors.T.dot(gene_values[self.gene_factor_gene_mask]) / self.exp_gene_factors.shape[0]
#     else:
#         self.factor_gene_scores = self.exp_lambdak


#     reorder_inds = np.argsort(-self.factor_gene_set_scores)
#     self.exp_lambdak = self.exp_lambdak[reorder_inds]
#     self.factor_gene_set_scores = self.factor_gene_set_scores[reorder_inds]
#     self.factor_gene_scores = self.factor_gene_scores[reorder_inds]
#     self.exp_gene_factors = self.exp_gene_factors[:,reorder_inds]
#     self.exp_gene_set_factors = self.exp_gene_set_factors[:,reorder_inds]

#     #now label them
#     gene_set_factor_gene_set_inds = np.where(self.gene_set_factor_gene_set_mask)[0]
#     gene_factor_gene_inds = np.where(self.gene_factor_gene_mask)[0]

#     num_top = 5
#     top_gene_inds = np.argsort(-self.exp_gene_factors, axis=0)[:num_top,:]
#     top_gene_set_inds = np.argsort(-self.exp_gene_set_factors, axis=0)[:num_top,:]

#     self.factor_labels = []
#     self.top_gene_sets = []
#     self.top_genes = []
#     factor_prompts = []
#     for i in range(len(self.factor_gene_set_scores)):
#         self.top_gene_sets.append([self.gene_sets[i] for i in np.where(self.gene_set_factor_gene_set_mask)[0][top_gene_set_inds[:,i]]])
#         self.top_genes.append([self.genes[i] for i in np.where(self.gene_factor_gene_mask)[0][top_gene_inds[:,i]]])
#         self.factor_labels.append(self.top_gene_sets[i][0] if len(self.top_gene_sets[i]) > 0 else "")
#         factor_prompts.append(",".join(self.top_gene_sets[i]))

#     if lmm_auth_key is not None:
#         prompt = "Print a label to assign to each group: %s" % (" ".join(["%d. %s" % (j+1, ",".join(self.top_gene_sets[j])) for j in range(len(self.top_gene_sets))]))
#         log("Querying LMM with prompt: %s" % prompt, DEBUG)
#         response = query_lmm(prompt, lmm_auth_key)
#         if response is not None:
#             try:
#                 responses = response.strip().split("\n")
#                 responses = [x for x in responses if len(x) > 0]
#                 if len(responses) == len(self.factor_labels):
#                     for i in range(len(self.factor_labels)):
#                         self.factor_labels[i] = responses[i]
#                 else:
#                     raise Exception
#             except Exception:
#                 log("Couldn't decode LMM response %s; using simple label" % response)
#                 pass


#     logger.info("Found %d factors" % len(self.exp_lambdak))
