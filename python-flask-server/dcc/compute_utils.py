
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

import dcc.dcc_utils as dutils 


# constants
logger = dutils.get_logger(__name__)


BATCH_SIZE = 4500

class RunFactorException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# methods
def compute_beta_tildes(X, Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=None, log_fun=log):

    log_fun("Calculating beta tildes")

    # Y can be a matrix with dimensions:
    # number of parallel runs x number of gene sets
    if len(Y.shape) == 2:
        len_Y = Y.shape[1]
        Y = (Y.T - np.mean(Y, axis=1)).T
    else:
        len_Y = Y.shape[0]
        Y = Y - np.mean(Y)

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
        # since variances are also in units of unscaled X, these will cancel out

        r_X = resid_correlation_matrix.dot(X)
        r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
        cor_variances = r_X_col_means - np.square(r_X_col_means)
        
        # never increase significance
        cor_variances[cor_variances < variances] = variances[cor_variances < variances]

        # both cor_variances and variances are in units of unscaled X
        se_inflation_factors = np.sqrt(cor_variances / variances)

    return finalize_regression(beta_tildes=beta_tildes, ses=ses, se_inflation_factors=se_inflation_factors)

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

    z_scores = np.zeros(beta_tildes.shape)_get_X_blocks_internal
    ses_positive_mask = ses > 0
    z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
    if np.any(~ses_positive_mask):
        logger.info("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))

    p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))

    # pvalues is what I want
    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)



#this code is adapted from https://github.com/gwas-partitioning/bnmf-clustering
def _bayes_nmf_l2(V0, n_iter=10000, a0=10, tol=1e-7, K=15, K0=15, phi=1.0):
    '''
    example?
        result = _bayes_nmf_l2(matrix, a0=alpha0, K=max_num_factors, K0=max_num_factors)
        exp_lambdak = result[4]
        exp_gene_factors = result[1].T
        exp_gene_set_factors = result[0]
    '''

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
            logger.info("Iteration=%d; evid=%.3g; lik=%.3g; err=%.3g; delambda=%.3g; factors=%d; factors_non_zero=%d" % (it, n_evid[it], n_like[it], n_error[it], delambda, np.sum(np.sum(W, axis=0) != 0), np.sum(lambdak >= lambda_cut)), TRACE)
        it += 1

    return W, H, n_like[-1], n_evid[-1], n_lambda[-1], n_error[-1]
        #W # Variant weight matrix (N x K)
        #H # Trait weight matrix (K x M)
        #n_like # List of reconstruction errors (sum of squared errors / 2) per iteration
        #n_evid # List of negative log-likelihoods per iteration
        #n_lambda # List of lambda vectors (shared weights for each of K clusters, some ~0) per iteration
        #n_error # List of reconstruction errors (sum of squared errors) per iteration


def _calc_X_shift_scale(X, y_corr_cholesky=None):
    '''
    returns the mean shifts and scale factors for the initial pathwayxgene matrix X
    '''
    if y_corr_cholesky is None:
        mean_shifts = X.sum(axis=0).A1 / X.shape[0]
        scale_factors = np.sqrt(X.power(2).sum(axis=0).A1 / X.shape[0] - np.square(mean_shifts))
    else:
        scale_factors = np.array([])
        mean_shifts = np.array([])
        for X_b, begin, end, batch in _get_X_blocks_internal(X, y_corr_cholesky):
            (cur_mean_shifts, cur_scale_factors) = _calc_shift_scale(X_b)
            mean_shifts = np.append(mean_shifts, cur_mean_shifts)
            scale_factors = np.append(scale_factors, cur_scale_factors)
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


# main
if __name__ == "__main__":
    pass








# old

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