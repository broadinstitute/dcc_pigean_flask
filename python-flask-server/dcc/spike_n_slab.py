"""
__author__ = "Jason Flannick and Edgar Llamas"
__copyright__ = "Copyright 2007, The 3 amigos Project"
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Edgar Llamas"
__email__ = "edgar.llamas@broadinstitute.org"
__status__ = "Production"
"""

import copy
import scipy
import scipy.sparse as sparse
import scipy.stats
import numpy as np
from numpy.random import gamma
import random
import sys
from scipy.optimize import fsolve

INFO = 1
DEBUG = 2
TRACE = 3


def learn_beta(x, data, coeff, p):
    return (1 / (1 + np.exp(-(data.dot(np.append(x.reshape((1, 1)), coeff, axis=0)))))).mean() - p


class SnS:

    def __init__(self, log_level=INFO):
        self.log_level = log_level

    def log(self, message, level=TRACE, end_char='\n'):
        log_fh = sys.stdout
        if level <= self.log_level:
            log_fh.write("%s%s" % (message, end_char))
            log_fh.flush()

    def calc_coefficients(self, logit_data, logit_y, wg_p, sigma_reg=0.001, assume_independent=False, correlation=None,
                          scale_factors=None, mean_shifts=None):
        num_features = logit_data.shape[1]
        log_sns_p = np.zeros(num_features)
        log_sns_coeff = np.zeros(num_features)
        prune_val = 0.8
        log_coeff = self.compute_logistic_beta_tildes(logit_data, logit_y)
        features_to_keep = self.prune_gene_sets(logit_data, log_coeff[3], prune_value=prune_val)
        if correlation is None:
            aux1, aux2 = \
                self.calculate_non_inf_betas(
                    log_coeff[0][:, features_to_keep],
                    log_coeff[1][:, features_to_keep],
                    X_orig=logit_data[:, features_to_keep],
                    sigma2=np.ones((1, 1)) * sigma_reg,
                    p=np.ones((1, 1)) * 0.001, assume_independent=assume_independent)
        else:
            aux1, aux2 = \
                self.calculate_non_inf_betas(
                    log_coeff[0][:, features_to_keep],
                    log_coeff[1][:, features_to_keep],
                    sigma2=np.ones((1, 1)) * sigma_reg,
                    p=np.ones((1, 1)) * 0.001, V=correlation[:, features_to_keep][features_to_keep, :],
                    scale_factors=scale_factors, mean_shifts=mean_shifts,
                    assume_independent=assume_independent)

        log_sns_coeff[features_to_keep] = aux1
        log_sns_p[features_to_keep] = aux2

        # Calculation of the intercept
        ext_data = np.append(np.ones((logit_data.shape[0], 1)), logit_data, 1)
        wg_coeff = log_sns_coeff.reshape((-1, 1))
        x0 = -np.log((1 - wg_p.mean()) / wg_p.mean())
        new_beta = fsolve(learn_beta, x0, (ext_data, wg_coeff, wg_p.mean())).reshape((1, 1))
        return log_coeff, new_beta, wg_coeff, log_sns_p

    def prune_gene_sets(self, X_orig, p_values, prune_value, max_size=5000, keep_missing=False, ignore_missing=False, skip_V=False):
        # modified variables ===================================
        scale_factors = X_orig.std(axis=0)
        mean_shifts = X_orig.mean(axis=0)
        # gene_set_masks = self._compute_gene_set_batches(X_orig=X_orig, mean_shifts=mean_shifts,
        #                                                 scale_factors=scale_factors, use_sum=True)
        gene_sets = list(range(p_values.shape[1]))
        # ======================================================
        keep_mask = np.array([False] * len(gene_sets))
        remove_gene_sets = set()

        #keep total to batch_size ** 2

        batch_size = int(max_size ** 2 / X_orig.shape[1])
        num_batches = int(X_orig.shape[1] / batch_size) + 1

        for batch in range(num_batches):
            begin = batch * batch_size
            end = (batch + 1) * batch_size
            if end > X_orig.shape[1]:
                end = X_orig.shape[1]

            X_b1  = X_orig[:,begin:end]

            V_block = self._compute_V(X_orig[:,begin:end], mean_shifts[begin:end], scale_factors[begin:end], X_orig2=X_orig, mean_shifts2=mean_shifts, scale_factors2=scale_factors)

            if p_values is not None and False:
                gene_set_key = lambda i: p_values[i]
            else:
                gene_set_key = lambda i: np.abs(X_b1[:,i]).sum(axis=0)

            for gene_set_ind in sorted(range(len(gene_sets[begin:end])), key=gene_set_key):
                absolute_ind = gene_set_ind + begin
                if absolute_ind in remove_gene_sets:
                    continue
                keep_mask[absolute_ind] = True
                remove_gene_sets.update(np.where(np.abs(V_block[gene_set_ind,:]) > prune_value)[0])
        self.log("Pruning at %.3g resulted in %d gene sets" % (prune_value, len(gene_sets)))
        return keep_mask

    def _compute_V(self, X_orig, mean_shifts, scale_factors, rows=None, X_orig2=None, mean_shifts2=None,
                   scale_factors2=None):
        if X_orig2 is None:
            X_orig2 = X_orig
        if mean_shifts2 is None:
            mean_shifts2 = mean_shifts
        if scale_factors2 is None:
            scale_factors2 = scale_factors
        if rows is None:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig.T.dot(X_orig2)
            else:
                dot_product = X_orig.T.dot(X_orig2).toarray().astype(float)
        else:
            if type(X_orig) is np.ndarray or type(X_orig2) is np.ndarray:
                dot_product = X_orig[:, rows].T.dot(X_orig2)
            else:
                dot_product = X_orig[:, rows].T.dot(X_orig2).toarray().astype(float)
            mean_shifts = mean_shifts[rows]
            scale_factors = scale_factors[rows]

        return (dot_product / X_orig.shape[0] - np.outer(mean_shifts, mean_shifts2)) / np.outer(scale_factors,
                                                                                                scale_factors2)

    def _compute_gene_set_batches(self, V=None, X_orig=None, mean_shifts=None, scale_factors=None, use_sum=True,
                                  max_allowed_batch_correlation=None, find_correlated_instead=None, sort_values=None,
                                  stop_at=None):
        batch_size = 4500
        gene_set_masks = []

        if max_allowed_batch_correlation is None:
            if use_sum:
                max_allowed_batch_correlation = 0.5
            else:
                max_allowed_batch_correlation = 0.1

        if find_correlated_instead is not None:
            if find_correlated_instead < 1:
                self.log("Need batch size of at least 1")

        if use_sum:
            combo_fn = np.sum
        else:
            combo_fn = np.max

        use_X = False
        if V is not None and len(V.shape) == 3:
            num_gene_sets = V.shape[1]
            not_included_gene_sets = np.full((V.shape[0], num_gene_sets), True)
        elif V is not None:
            num_gene_sets = V.shape[0]
            not_included_gene_sets = np.full(num_gene_sets, True)
        else:
            assert (mean_shifts.shape == scale_factors.shape)
            if len(mean_shifts.shape) > 1:
                mean_shifts = np.squeeze(mean_shifts, axis=0)
                scale_factors = np.squeeze(scale_factors, axis=0)
            if X_orig is None or mean_shifts is None or scale_factors is None:
                self.log("Need X_orig or V for this operation")
            num_gene_sets = X_orig.shape[1]
            not_included_gene_sets = np.full(num_gene_sets, True)
            use_X = True

        self.log("Batching %d gene sets..." % num_gene_sets)
        if use_X:
            self.log("Using low memory mode")

        indices = np.array(range(num_gene_sets))

        if sort_values is None:
            sort_values = indices

        total_added = 0

        while np.any(not_included_gene_sets):
            if V is not None and len(V.shape) == 3:
                # batches if multiple_V

                current_mask = np.full((V.shape[0], num_gene_sets), False)
                # set the first gene set in each row to True
                for c in range(V.shape[0]):

                    sorted_remaining_indices = sorted(indices[not_included_gene_sets[c, :]], key=lambda k: sort_values[k])
                    # seed with the first gene not already included
                    if len(sorted_remaining_indices) == 0:
                        continue

                    first_gene_set = sorted_remaining_indices[0]
                    current_mask[c, first_gene_set] = True
                    not_included_gene_sets[c, first_gene_set] = False
                    sorted_remaining_indices = sorted_remaining_indices[1:]

                    if find_correlated_instead:
                        # WARNING: THIS HAS NOT BEEN TESTED
                        # sort by decreasing V
                        index_map = np.where(not_included_gene_sets[c, :])[0]
                        ordered_indices = index_map[np.argsort(-V[c, first_gene_set, :])[not_included_gene_sets[c, :]]]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[c, indices_to_add] = True
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[c, i, current_mask[c, :]]) < max_allowed_batch_correlation:
                                current_mask[c, i] = True
                                not_included_gene_sets[c, i] = False
            else:
                sorted_remaining_indices = sorted(indices[not_included_gene_sets], key=lambda k: sort_values[k])
                # batches if one V
                current_mask = np.full(num_gene_sets, False)
                # seed with the first gene not already included
                first_gene_set = sorted_remaining_indices[0]
                current_mask[first_gene_set] = True
                not_included_gene_sets[first_gene_set] = False
                sorted_remaining_indices = sorted_remaining_indices[1:]

                if V is not None:
                    if find_correlated_instead:
                        # sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-V[first_gene_set, not_included_gene_sets])]
                        # map these to the original ones
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        for i in sorted_remaining_indices:
                            if combo_fn(V[i, current_mask]) < max_allowed_batch_correlation:
                                current_mask[i] = True
                                not_included_gene_sets[i] = False
                else:
                    assert (scale_factors.shape == mean_shifts.shape)

                    if find_correlated_instead:
                        cur_V = self._compute_V(X_orig[:, first_gene_set], mean_shifts[first_gene_set],
                                                scale_factors[first_gene_set], X_orig2=X_orig[:, not_included_gene_sets],
                                                mean_shifts2=mean_shifts[not_included_gene_sets],
                                                scale_factors2=scale_factors[not_included_gene_sets])
                        # sort by decreasing V
                        index_map = np.where(not_included_gene_sets)[0]
                        ordered_indices = index_map[np.argsort(-cur_V[0, :])]
                        indices_to_add = ordered_indices[:find_correlated_instead]
                        current_mask[indices_to_add] = True
                        not_included_gene_sets[indices_to_add] = False
                    else:
                        # cap out at batch_size gene sets to avoid memory of making whole V; this may reduce the batch size relative to optimal
                        # also, only add those not in mask already (since we are searching only these in V)
                        max_to_add = batch_size
                        V_to_generate_mask = copy.copy(not_included_gene_sets)
                        if np.sum(V_to_generate_mask) > max_to_add:
                            assert (len(sorted_remaining_indices) == np.sum(not_included_gene_sets))
                            V_to_generate_mask[sort_values > sort_values[sorted_remaining_indices[max_to_add]]] = False

                        V_to_generate_mask[first_gene_set] = True
                        cur_V = self._compute_V(X_orig[:, V_to_generate_mask], mean_shifts[V_to_generate_mask],
                                                scale_factors[V_to_generate_mask])
                        indices_not_included = indices[V_to_generate_mask]
                        sorted_cur_V_indices = sorted(range(cur_V.shape[0]),
                                                      key=lambda k: sort_values[indices_not_included[k]])
                        for i in sorted_cur_V_indices:
                            if combo_fn(cur_V[i, current_mask[V_to_generate_mask]]) < max_allowed_batch_correlation:
                                current_mask[indices_not_included[i]] = True
                                not_included_gene_sets[indices_not_included[i]] = False

            gene_set_masks.append(current_mask)
            # self.log("Batch %d; %d gene sets" % (len(gene_set_masks), sum(current_mask)), TRACE)
            total_added += np.sum(current_mask)
            if stop_at is not None and total_added >= stop_at:
                self.log("Breaking at %d" % total_added)
                break

        denom = 1
        if V is not None and len(V.shape) == 3:
            denom = V.shape[0]

        sizes = [float(np.sum(x)) / denom for x in gene_set_masks]
        self.log("Batched %d gene sets into %d batches; size range %d - %d" % (
        num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes) if len(sizes) > 0 else 0))

        return gene_set_masks

    def get_scaled_sigma2(self, scale_factors, sigma2, sigma_power, sigma_threshold_k, sigma_threshold_xo):
        threshold = 1
        if sigma_threshold_k is not None and sigma_threshold_xo is not None:
            threshold =  1 / (1 + np.exp(-sigma_threshold_k * (scale_factors - sigma_threshold_xo)))
        return threshold * sigma2 * np.power(scale_factors, sigma_power)

    def calculate_non_inf_betas(self, beta_tildes, ses, X_orig=None, sigma2=None, sigma2s=None, ps=None,
                                p=None, initial_p=None,
                                return_sample=False, max_num_burn_in=None,
                                max_num_iter=1100,
                                min_num_iter=10, num_chains=2, r_threshold_burn_in=1.01,
                                use_max_r_for_convergence=True, eps=0.01,
                                max_allowed_batch_correlation=None, beta_outlier_iqr_threshold=5, gauss_seidel=False,
                                update_hyper_sigma=True, update_hyper_p=True, adjust_hyper_sigma_p=False,
                                sigma_num_devs_to_top=2.0, p_noninf_inflate=1.0, num_p_pseudo=1, sparse_solution=False,
                                sparse_frac_betas=None, betas_trace_out=None, betas_trace_gene_sets=None,
                                V=None, scale_factors=None, mean_shifts=None, assume_independent=False,
                                num_missing_gene_sets=None):
        """
        #param: p one per chain
        """

        debug_gene_sets = None

        if max_num_burn_in is None:
            max_num_burn_in = int(max_num_iter * .25)
        if max_num_burn_in >= max_num_iter:
            max_num_burn_in = int(max_num_iter * .25)

        # if (update_hyper_p or update_hyper_sigma) and gauss_seidel:
        #    self.log("Using Gibbs sampling for betas since update hyper was requested")
        #    gauss_seidel = False
        if X_orig is not None:
            scale_factors = X_orig.std(axis=0)
            mean_shifts = X_orig.mean(axis=0)

        use_X = False
        if V is None and not assume_independent:
            if X_orig is None or scale_factors is None or mean_shifts is None:
                self.log("Require X, scale, and mean if V is None")
            else:
                use_X = True
                self.log("Using low memory X instead of V")

        if sigma2 is None:
            self.log("Need sigma to calculate betas!")

        if p is None and ps is None:
            self.log("Need p to calculate non-inf betas")

        if not len(beta_tildes.shape) == len(ses.shape):
            self.log("If running parallel beta inference, beta_tildes and ses must have same shape")

        if len(beta_tildes.shape) == 0 or beta_tildes.shape[0] == 0:
            self.log("No gene sets are left!")

        # convert the beta_tildes and ses to matrices -- columns are num_parallel
        # they are always stored as matrices, with 1 column as needed
        # V on the other hand will be a 2-D matrix if it is constant across all parallel (or if there is only 1)
        # checking len(V.shape) can therefore distinguish a constant from variable V

        multiple_V = False
        sparse_V = False

        if len(beta_tildes.shape) > 1:
            num_gene_sets = beta_tildes.shape[1]

            if not beta_tildes.shape[0] == ses.shape[0]:
                self.log("beta_tildes and ses must have same number of parallel runs")

            # dimensions should be num_gene_sets, num_parallel
            num_parallel = beta_tildes.shape[0]
            beta_tildes_m = copy.copy(beta_tildes)
            ses_m = copy.copy(ses)

            if V is not None and type(V) is sparse.csc_matrix:
                sparse_V = True
                multiple_V = False
            elif V is not None and len(V.shape) == 3:
                if not V.shape[0] == beta_tildes.shape[0]:
                    self.log("V must have same number of parallel runs as beta_tildes")
                multiple_V = True
                sparse_V = False
            else:
                multiple_V = False
                sparse_V = False
        else:
            num_gene_sets = len(beta_tildes)
            if V is not None and type(V) is sparse.csc_matrix:
                num_parallel = 1
                multiple_V = False
                sparse_V = True
                beta_tildes_m = beta_tildes[np.newaxis, :]
                ses_m = ses[np.newaxis, :]
            elif V is not None and len(V.shape) == 3:
                num_parallel = V.shape[0]
                multiple_V = True
                sparse_V = False
                beta_tildes_m = np.tile(beta_tildes, num_parallel).reshape((num_parallel, len(beta_tildes)))
                ses_m = np.tile(ses, num_parallel).reshape((num_parallel, len(ses)))
            else:
                num_parallel = 1
                multiple_V = False
                sparse_V = False
                beta_tildes_m = beta_tildes[np.newaxis, :]
                ses_m = ses[np.newaxis, :]

        if num_parallel == 1 and multiple_V:
            multiple_V = False
            V = V[0, :, :]

        if multiple_V:
            assert (not use_X)

        if scale_factors.shape != mean_shifts.shape:
            self.log("scale_factors must have same dimension as mean_shifts")

        if len(scale_factors.shape) == 2 and not scale_factors.shape[0] == num_parallel:
            self.log("scale_factors must have same number of parallel runs as beta_tildes")
        elif len(scale_factors.shape) == 1 and num_parallel == 1:
            scale_factors_m = scale_factors[np.newaxis, :]
            mean_shifts_m = mean_shifts[np.newaxis, :]
        elif len(scale_factors.shape) == 1 and num_parallel > 1:
            scale_factors_m = np.tile(scale_factors, num_parallel).reshape((num_parallel, len(scale_factors)))
            mean_shifts_m = np.tile(mean_shifts, num_parallel).reshape((num_parallel, len(mean_shifts)))
        else:
            scale_factors_m = copy.copy(scale_factors)
            mean_shifts_m = copy.copy(mean_shifts)

        if ps is not None:
            if len(ps.shape) == 2 and not ps.shape[0] == num_parallel:
                self.log("ps must have same number of parallel runs as beta_tildes")
            elif len(ps.shape) == 1 and num_parallel == 1:
                ps_m = ps[np.newaxis, :]
            elif len(ps.shape) == 1 and num_parallel > 1:
                ps_m = np.tile(ps, num_parallel).reshape((num_parallel, len(ps)))
            else:
                ps_m = copy.copy(ps)
        else:
            ps_m = p

        if sigma2s is not None:
            if len(sigma2s.shape) == 2 and not sigma2s.shape[0] == num_parallel:
                self.log("sigma2s must have same number of parallel runs as beta_tildes")
            elif len(sigma2s.shape) == 1 and num_parallel == 1:
                orig_sigma2_m = sigma2s[np.newaxis, :]
            elif len(sigma2s.shape) == 1 and num_parallel > 1:
                orig_sigma2_m = np.tile(sigma2s, num_parallel).reshape((num_parallel, len(sigma2s)))
            else:
                orig_sigma2_m = copy.copy(sigma2s)
        else:
            orig_sigma2_m = sigma2

        # for efficiency, batch genes to be updated each cycle
        if assume_independent:
            gene_set_masks = [np.full(beta_tildes_m.shape[1], True)]
        else:
            gene_set_masks = self._compute_gene_set_batches(V, X_orig=X_orig, mean_shifts=mean_shifts,
                                                            scale_factors=scale_factors, use_sum=True,
                                                            max_allowed_batch_correlation=max_allowed_batch_correlation)

        sizes = [float(np.sum(x)) / (num_parallel if multiple_V else 1) for x in gene_set_masks]
        self.log("Analyzing %d gene sets in %d batches of gene sets; size range %d - %d" % (
            num_gene_sets, len(gene_set_masks), min(sizes) if len(sizes) > 0 else 0, max(sizes) if len(sizes) > 0 else 0))

        # get the dimensions of the gene_set_masks to match those of the betas
        if num_parallel == 1:
            assert (not multiple_V)
            # convert the vectors into matrices with one dimension
            gene_set_masks = [x[np.newaxis, :] for x in gene_set_masks]
        elif not multiple_V:
            # we have multiple parallel but only one V
            gene_set_masks = [np.tile(x, num_parallel).reshape((num_parallel, len(x))) for x in gene_set_masks]

        # variables are denoted
        # v: vectors of dimension equal to the number of gene sets
        # m: data that varies by parallel runs and gene sets
        # t: data that varies by chains, parallel runs, and gene sets

        # rules:
        # 1. adding a lower dimensional tensor to higher dimenional ones means final dimensions must match. These operations are usually across replicates
        # 2. lower dimensional masks on the other hand index from the beginning dimensions (can use :,:,mask to index from end)

        tensor_shape = (num_chains, num_parallel, num_gene_sets)
        matrix_shape = (num_parallel, num_gene_sets)

        # these are current posterior means (including p and the conditional beta). They are used to calculate avg_betas
        # using these as the actual betas would yield the Gauss-seidel algorithm
        curr_post_means_t = np.zeros(tensor_shape)
        curr_postp_t = np.ones(tensor_shape)

        # these are the current betas to be used in each iteration
        initial_sd = np.std(beta_tildes_m)
        if initial_sd == 0:
            initial_sd = 1

        curr_betas_t = scipy.stats.norm.rvs(0, initial_sd, tensor_shape)

        res_beta_hat_t = np.zeros(tensor_shape)

        avg_betas_m = np.zeros(matrix_shape)
        avg_betas2_m = np.zeros(matrix_shape)
        avg_postp_m = np.zeros(matrix_shape)
        num_avg = 0

        # these are the posterior betas averaged across iterations
        sum_betas_t = np.zeros(tensor_shape)
        sum_betas2_t = np.zeros(tensor_shape)

        # Setting up constants
        # hyperparameters
        # shrinkage prior
        sigma_power = 2

        tag = ""
        if assume_independent:
            tag = "independent "
        elif sparse_V:
            tag = "partially independent "

        self.log("Calculating %snon-infinitesimal betas with %s, %s; %s" % (tag, p, sigma2, sigma2))

        # generate the diagonals to use per replicate
        if assume_independent:
            V_diag_m = None
            account_for_V_diag_m = False
        else:
            if V is not None:
                if num_parallel > 1:
                    # dimensions are num_parallel, num_gene_sets, num_gene_sets
                    if multiple_V:
                        V_diag_m = np.diagonal(V, axis1=1, axis2=2)
                    else:
                        if sparse_V:
                            V_diag = V.diagonal()
                        else:
                            V_diag = np.diag(V)
                        V_diag_m = np.tile(V_diag, num_parallel).reshape((num_parallel, len(V_diag)))
                else:
                    if sparse_V:
                        V_diag_m = V.diagonal()[np.newaxis, :]
                    else:
                        V_diag_m = np.diag(V)[np.newaxis, :]

                account_for_V_diag_m = not np.isclose(V_diag_m, np.ones(matrix_shape)).all()
            else:
                # we compute it from X, so we know it is always 1
                V_diag_m = None
                account_for_V_diag_m = False

        se2s_m = np.power(ses_m, 2)

        # the below code is based off of the LD-pred code for SNP PRS
        iteration_num = 0
        burn_in_phase_v = np.array([True for i in range(num_parallel)])

        # self.log("TEST V_DIAG_M!")
        prev_betas_m = None
        sigma_underflow = False
        printed_warning_swing = False
        printed_warning_increase = False
        while iteration_num < max_num_iter:  # Big iteration

            # if some have not converged, only sample for those that have not converged (for efficiency)
            compute_mask_v = copy.copy(burn_in_phase_v)
            if np.sum(compute_mask_v) == 0:
                compute_mask_v[:] = True

            hdmp_m = (sigma2 / ps_m)
            hdmpn_m = hdmp_m + se2s_m
            hdmp_hdmpn_m = (hdmp_m / hdmpn_m)

            # if iteration_num == 0:
            #    self.log(hdmp_hdmpn_m[0,:],(sigma2_m / ps_m)[0,:],se2s_m[0,:])

            norm_scale_m = np.sqrt(np.multiply(hdmp_hdmpn_m, se2s_m))
            c_const_m = (ps_m / np.sqrt(hdmpn_m))

            d_const_m = (1 - ps_m) / ses_m

            iteration_num += 1

            # default to 1
            curr_postp_t[:, compute_mask_v, :] = np.ones(tensor_shape)[:, compute_mask_v, :]

            # sample whether each gene set has non-zero effect
            rand_ps_t = np.random.random(tensor_shape)
            # generate normal random variable sampling
            rand_norms_t = scipy.stats.norm.rvs(0, 1, tensor_shape)

            for gene_set_mask_ind in range(len(gene_set_masks)):

                # the challenge here is that gene_set_mask_m produces a ragged (non-square) tensor
                # so we are going to "flatten" the last two dimensions
                # this requires some care, in particular when running einsum, which requires a square tensor

                gene_set_mask_m = gene_set_masks[gene_set_mask_ind]

                if debug_gene_sets is not None:
                    cur_debug_gene_sets = [debug_gene_sets[i] for i in range(len(debug_gene_sets)) if
                                           gene_set_mask_m[0, i]]

                # intersect compute_max_v with the rows of gene_set_mask (which are the parallel runs)
                compute_mask_m = np.logical_and(compute_mask_v, gene_set_mask_m.T).T

                current_num_parallel = sum(compute_mask_v)

                alpha_shrink = 1

                # subtract out the predicted effects of the other betas
                # we need to zero out diagonal of V to do this, but rather than do this we will add it back in

                # 1. First take the union of the current_gene_set_mask
                # this is to allow us to run einsum
                # we are going to do it across more gene sets than are needed, and then throw away the computations that are extra for each batch
                compute_mask_union = np.any(compute_mask_m, axis=0)

                # 2. Retain how to filter from the union down to each mask
                compute_mask_union_filter_m = compute_mask_m[:, compute_mask_union]

                if assume_independent:
                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m]
                else:
                    if multiple_V:

                        # 3. Do einsum across the union
                        # This does pointwise matrix multiplication of curr_betas_t (sliced on axis 1) with V (sliced on axis 0), maintaining axis 0 for curr_betas_t
                        res_beta_hat_union_t = np.einsum('hij,ijk->hik', curr_betas_t[:, compute_mask_v, :],
                                                         V[compute_mask_v, :, :][:, :, compute_mask_union]).reshape(
                            (num_chains, current_num_parallel, np.sum(compute_mask_union)))

                    elif sparse_V:
                        res_beta_hat_union_t = V[compute_mask_union, :].dot(
                            curr_betas_t[:, compute_mask_v, :].T.reshape(
                                (curr_betas_t.shape[2], np.sum(compute_mask_v) * curr_betas_t.shape[0]))).reshape(
                            (np.sum(compute_mask_union), np.sum(compute_mask_v), curr_betas_t.shape[0])).T
                    elif use_X:
                        if len(compute_mask_union.shape) == 2:
                            assert (compute_mask_union.shape[0] == 1)
                            compute_mask_union = np.squeeze(compute_mask_union)
                        # curr_betas_t: (num_chains, num_parallel, num_gene_sets)
                        # X_orig: (num_genes, num_gene_sets)
                        # X_orig_t: (num_gene_sets, num_genes)
                        # mean_shifts_m: (num_parallel, num_gene_sets)

                        curr_betas_filtered_t = curr_betas_t[:, compute_mask_v, :] / scale_factors_m[compute_mask_v, :]

                        # have to reshape latter two dimensions before multiplying because sparse matrix can only handle 2-D
                        # todo: fix
                        interm = np.zeros((X_orig.shape[0], curr_betas_t.shape[1], curr_betas_t.shape[0]))
                        # interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],
                        #                                                      curr_betas_filtered_t.shape[0] *
                        #                                                      curr_betas_filtered_t.shape[1]))).reshape(
                        #     (X_orig.shape[0], curr_betas_t.shape[1], curr_betas_t.shape[0])) - np.sum(
                        #     mean_shifts_m * curr_betas_filtered_t, axis=2).T
                        interm = X_orig.dot(curr_betas_filtered_t.T.reshape((curr_betas_filtered_t.shape[2],
                                                                             curr_betas_filtered_t.shape[0] *
                                                                             curr_betas_filtered_t.shape[1]))).reshape(
                            (X_orig.shape[0], curr_betas_filtered_t.shape[1], curr_betas_filtered_t.shape[0])) - np.sum(mean_shifts_m[compute_mask_v, :] * curr_betas_filtered_t, axis=2).T
                        # interm: (num_genes, num_parallel, num_chains)

                        # num_gene sets, num_parallel, num_chains
                        res_beta_hat_union_t = (X_orig[:, compute_mask_union].T.dot(
                            interm.reshape((interm.shape[0], interm.shape[1] * interm.shape[2]))).reshape(
                            (np.sum(compute_mask_union), interm.shape[1], interm.shape[2])) - mean_shifts_m.T[
                                                                                              compute_mask_union, :][:,
                                                                                              compute_mask_v,
                                                                                              np.newaxis] * np.sum(
                            interm, axis=0)).T

                        res_beta_hat_union_t /= (
                                X_orig.shape[0] * scale_factors_m[compute_mask_v, :][:, compute_mask_union])
                    else:
                        res_beta_hat_union_t = curr_betas_t[:, compute_mask_v, :].dot(V[:, compute_mask_union])


                    # 4. Now restrict to only the actual masks (which flattens things because the compute_mask_m is not square)

                    res_beta_hat_t_flat = res_beta_hat_union_t[:, compute_mask_union_filter_m[compute_mask_v, :]]
                    assert (res_beta_hat_t_flat.shape[1] == np.sum(compute_mask_m))

                    # dimensions of res_beta_hat_t_flat are (num_chains, np.sum(compute_mask_m))
                    # dimensions of beta_tildes_m are (num_parallel, num_gene_sets))
                    # subtraction will subtract matrix from each of the matrices in the tensor

                    res_beta_hat_t_flat = beta_tildes_m[compute_mask_m] - res_beta_hat_t_flat

                    if account_for_V_diag_m:
                        # dimensions of V_diag_m are (num_parallel, num_gene_sets)
                        # curr_betas_t is (num_chains, num_parallel, num_gene_sets)
                        res_beta_hat_t_flat = res_beta_hat_t_flat + V_diag_m[compute_mask_m] * curr_betas_t[:,
                                                                                               compute_mask_m]
                    else:
                        res_beta_hat_t_flat = res_beta_hat_t_flat + curr_betas_t[:, compute_mask_m]

                b2_t_flat = np.power(res_beta_hat_t_flat, 2)
                d_const_b2_exp_t_flat = d_const_m[compute_mask_m] * np.exp(-b2_t_flat / (se2s_m[compute_mask_m] * 2.0))
                numerator_t_flat = c_const_m[compute_mask_m] * np.exp(-b2_t_flat / (2.0 * hdmpn_m[compute_mask_m]))
                numerator_zero_mask_t_flat = (numerator_t_flat == 0)
                denominator_t_flat = numerator_t_flat + d_const_b2_exp_t_flat
                denominator_t_flat[numerator_zero_mask_t_flat] = 1

                d_imaginary_mask_t_flat = ~np.isreal(d_const_b2_exp_t_flat)
                numerator_imaginary_mask_t_flat = ~np.isreal(numerator_t_flat)

                if np.any(np.logical_or(d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)):
                    self.log("Detected imaginary numbers!")
                    # if d is imaginary, we set it to 1
                    denominator_t_flat[d_imaginary_mask_t_flat] = numerator_t_flat[d_imaginary_mask_t_flat]
                    # if d is real and numerator is imaginary, we set to 0 (both numerator and denominator will be imaginary)
                    numerator_t_flat[np.logical_and(~d_imaginary_mask_t_flat, numerator_imaginary_mask_t_flat)] = 0

                    # Original code for handling edge cases; adapted above
                    # Commenting these out for now, but they are here in case we ever detect non real numbers
                    # if need them, masked_array is too inefficient -- change to real mask
                    # d_real_mask_t = np.isreal(d_const_b2_exp_t)
                    # numerator_real_mask_t = np.isreal(numerator_t)
                    # curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_not(d_real_mask_t), fill_value = 1).filled()
                    # curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(d_real_mask_t, np.logical_not(numerator_real_mask_t)), fill_value=0).filled()
                    # curr_postp_t = np.ma.masked_array(curr_postp_t, np.logical_and(np.logical_and(d_real_mask_t, numerator_real_mask_t), numerator_zero_mask_t), fill_value=0).filled()

                curr_postp_t[:, compute_mask_m] = (numerator_t_flat / denominator_t_flat)

                # calculate current posterior means
                # the left hand side, because it is masked, flattens the latter two dimensions into one
                # so we flatten the result of the right hand size to a 1-D array to match up for the assignment
                curr_post_means_t[:, compute_mask_m] = hdmp_hdmpn_m[compute_mask_m] * (
                        curr_postp_t[:, compute_mask_m] * res_beta_hat_t_flat)

                if gauss_seidel:
                    proposed_beta_t_flat = curr_post_means_t[:, compute_mask_m]
                else:
                    norm_mean_t_flat = hdmp_hdmpn_m[compute_mask_m] * res_beta_hat_t_flat

                    # draw from the conditional distribution
                    proposed_beta_t_flat = norm_mean_t_flat + norm_scale_m[compute_mask_m] * rand_norms_t[:,
                                                                                             compute_mask_m]

                    # set things to zero that sampled below p
                    zero_mask_t_flat = rand_ps_t[:, compute_mask_m] >= curr_postp_t[:, compute_mask_m] * alpha_shrink
                    proposed_beta_t_flat[zero_mask_t_flat] = 0

                # update betas
                # do this inside loop since this determines the res_beta
                # same idea as above for collapsing
                curr_betas_t[:, compute_mask_m] = proposed_beta_t_flat
                res_beta_hat_t[:, compute_mask_m] = res_beta_hat_t_flat

            if sparse_solution:
                sparse_mask_t = curr_postp_t < ps_m

                if sparse_frac_betas is not None:
                    # zero out very small values relative to top or median
                    relative_value = np.max(np.abs(curr_post_means_t), axis=2)
                    sparse_mask_t = np.logical_or(sparse_mask_t, (
                            np.abs(curr_post_means_t).T < sparse_frac_betas * relative_value.T).T)

                # don't set anything not currently computed
                sparse_mask_t[:, np.printical_not(compute_mask_v), :] = False
                self.log("Setting %d entries to zero due to sparsity" % (
                    np.sum(np.logical_and(sparse_mask_t, curr_betas_t > 0))))
                curr_betas_t[sparse_mask_t] = 0
                curr_post_means_t[sparse_mask_t] = 0

            curr_betas_m = np.mean(curr_post_means_t, axis=0)
            curr_postp_m = np.mean(curr_postp_t, axis=0)
            # no state should be preserved across runs, but take a random one just in case
            sample_betas_m = curr_betas_t[int(random.random() * curr_betas_t.shape[0]), :, :]
            sample_postp_m = curr_postp_t[int(random.random() * curr_postp_t.shape[0]), :, :]
            sum_betas_t[:, compute_mask_v, :] = sum_betas_t[:, compute_mask_v, :] + curr_post_means_t[:, compute_mask_v,
                                                                                    :]
            sum_betas2_t[:, compute_mask_v, :] = sum_betas2_t[:, compute_mask_v, :] + np.square(
                curr_post_means_t[:, compute_mask_v, :])

            # now calculate the convergence metrics
            R_m = np.zeros(matrix_shape)
            beta_weights_m = np.zeros(matrix_shape)
            sem2_m = np.zeros(matrix_shape)
            will_break = False
            if assume_independent:
                burn_in_phase_v[:] = False
            elif gauss_seidel:
                if prev_betas_m is not None:
                    sum_diff = np.sum(np.abs(prev_betas_m - curr_betas_m))
                    sum_prev = np.sum(np.abs(prev_betas_m))
                    tot_diff = sum_diff / sum_prev
                    self.log("Iteration %d: gauss seidel difference = %.4g / %.4g = %.4g" % (
                        iteration_num + 1, sum_diff, sum_prev, tot_diff))
                    if iteration_num > min_num_iter and tot_diff < eps:
                        burn_in_phase_v[:] = False
                        self.log("Converged after %d iterations" % (iteration_num + 1))
                prev_betas_m = curr_betas_m
            elif iteration_num > min_num_iter and np.sum(burn_in_phase_v) > 0:
                def __calculate_R_tensor(sum_t, sum2_t, num, print_pc=None):

                    # mean of betas across all iterations; psi_dot_j
                    mean_t = sum_t / float(num)

                    if print_pc is not None:
                        self.log(mean_t[print_pc[1], print_pc[0], :10])

                    # mean of betas across replicates; psi_dot_dot
                    mean_m = np.mean(mean_t, axis=0)
                    # variances of betas across all iterators; s_j
                    var_t = (sum2_t - float(num) * np.power(mean_t, 2)) / (float(num) - 1)
                    # B_v = (float(iteration_num) / (num_chains - 1)) * np.apply_along_axis(np.sum, 0, np.apply_along_axis(lambda x: np.power(x - mean_betas_v, 2), 1, mean_betas_m))
                    B_m = (float(num) / (mean_t.shape[0] - 1)) * np.sum(np.power(mean_t - mean_m, 2), axis=0)
                    W_m = (1.0 / float(mean_t.shape[0])) * np.sum(var_t, axis=0)
                    avg_W_m = (1.0 / float(mean_t.shape[2])) * np.sum(var_t, axis=2)
                    var_given_y_m = np.add((float(num) - 1) / float(num) * W_m, (1.0 / float(num)) * B_m)
                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m = np.ones(W_m.shape)
                    R_non_zero_mask_m = W_m > 0

                    var_given_y_m[var_given_y_m < 0] = 0

                    R_m[R_non_zero_mask_m] = np.sqrt(var_given_y_m[R_non_zero_mask_m] / W_m[R_non_zero_mask_m])

                    return (B_m, W_m, R_m, avg_W_m, mean_t)

                # these matrices have convergence statistics in format (num_parallel, num_gene_sets)
                # WARNING: only the results for compute_mask_v are valid
                (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num)

                beta_weights_m = np.zeros((sum_betas_t.shape[1], sum_betas_t.shape[2]))
                sum_betas_t_mean = np.mean(sum_betas_t)
                if sum_betas_t_mean > 0:
                    np.mean(sum_betas_t, axis=0) / sum_betas_t_mean

                # calculate the thresholded / scaled R_v
                num_R_above_1_v = np.sum(R_m >= 1, axis=1)
                num_R_above_1_v[num_R_above_1_v == 0] = 1

                # mean for each parallel run

                R_m_above_1 = copy.copy(R_m)
                R_m_above_1[R_m_above_1 < 1] = 0
                mean_thresholded_R_v = np.sum(R_m_above_1, axis=1) / num_R_above_1_v

                # max for each parallel run
                max_index_v = np.argmax(R_m, axis=1)
                max_index_parallel = None
                max_val = None
                for i in range(len(max_index_v)):
                    if compute_mask_v[i] and (max_val is None or R_m[i, max_index_v[i]] > max_val):
                        max_val = R_m[i, max_index_v[i]]
                        max_index_parallel = i
                max_R_v = np.max(R_m, axis=1)

                # TEMP TEMP TEMP
                # if priors_for_convergence:
                #    curr_v = curr_betas_v
                #    s_cur2_v = np.array([curr_v[i] for i in sorted(range(len(curr_v)), key=lambda k: -np.abs(curr_v[k]))])
                #    s_cur2_v = np.square(s_cur2_v - np.mean(s_cur2_v))
                #    cum_cur2_v = np.cumsum(s_cur2_v) / np.sum(s_cur2_v)
                #    top_mask2 = np.array(cum_cur2_v < 0.99)
                #    (B_v2, W_v2, R_v2) = __calculate_R(sum_betas_m[:,top_mask2], sum_betas2_m[:,top_mask2], iteration_num)
                #    max_index2 = np.argmax(R_v2)
                #    self.log("Iteration %d (betas): max ind=%d; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g" % (iteration_num, max_index2, B_v2[max_index2], W_v2[max_index2], R_v2[max_index2], np.mean(R_v2), np.sum(R_v2 > r_threshold_burn_in)), TRACE)
                # END TEMP TEMP TEMP

                if use_max_r_for_convergence:
                    convergence_statistic_v = max_R_v
                else:
                    convergence_statistic_v = mean_thresholded_R_v

                outlier_mask_m = np.full(avg_W_m.shape, False)
                if avg_W_m.shape[0] > 10:
                    # check the variances
                    q3, median, q1 = np.percentile(avg_W_m, [75, 50, 25], axis=0)
                    iqr_mask = q3 > q1
                    chain_iqr_m = np.zeros(avg_W_m.shape)
                    chain_iqr_m[:, iqr_mask] = (avg_W_m[:, iqr_mask] - median[iqr_mask]) / (q3 - q1)[iqr_mask]
                    # dimensions chain x parallel
                    outlier_mask_m = beta_outlier_iqr_threshold
                    if np.sum(outlier_mask_m) > 0:
                        self.log("Detected %d outlier chains due to oscillations" % np.sum(outlier_mask_m))

                if np.sum(R_m > 1) > 10:
                    # check the Rs
                    q3, median, q1 = np.percentile(R_m[R_m > 1], [75, 50, 25])
                    if q3 > q1:
                        # Z score per parallel, gene
                        R_iqr_m = (R_m - median) / (q3 - q1)
                        # dimensions of parallel x gene sets
                        bad_gene_sets_m = np.logical_and(R_iqr_m > 100, R_m > 2.5)
                        bad_gene_sets_v = np.any(bad_gene_sets_m, 0)
                        if np.sum(bad_gene_sets_m) > 0:
                            # now find the bad chains
                            bad_chains = np.argmax(np.abs(mean_t - np.mean(mean_t, axis=0)), axis=0)[bad_gene_sets_m]

                            # np.where bad gene sets[0] lists parallel
                            # bad chains lists the bad chain corresponding to each parallel
                            cur_outlier_mask_m = np.zeros(outlier_mask_m.shape)
                            cur_outlier_mask_m[bad_chains, np.where(bad_gene_sets_m)[0]] = True

                            self.log("Found %d outlier chains across %d parallel runs due to %d gene sets with high R (%.4g - %.4g; %.4g - %.4g)" % (
                                np.sum(cur_outlier_mask_m), np.sum(np.any(cur_outlier_mask_m, axis=0)),
                                np.sum(bad_gene_sets_m), np.min(R_m[bad_gene_sets_m]), np.max(R_m[bad_gene_sets_m]),
                                np.min(R_iqr_m[bad_gene_sets_m]), np.max(R_iqr_m[bad_gene_sets_m])))
                            outlier_mask_m = np.logical_or(outlier_mask_m, cur_outlier_mask_m)

                            # self.log("Outlier parallel: %s" % (np.where(bad_gene_sets_m)[0]), DEBUG)
                            # self.log("Outlier values: %s" % (R_m[bad_gene_sets_m]), DEBUG)
                            # self.log("Outlier IQR: %s" % (R_iqr_m[bad_gene_sets_m]), DEBUG)
                            # self.log("Outlier chains: %s" % (bad_chains), DEBUG)

                            # self.log("Actually in mask: %s" % (str(np.where(outlier_mask_m))))

                non_outliers_m = ~outlier_mask_m
                if np.sum(outlier_mask_m) > 0:
                    self.log("Detected %d total outlier chains" % np.sum(outlier_mask_m))
                    # dimensions are num_chains x num_parallel
                    for outlier_parallel in np.where(np.any(outlier_mask_m, axis=0))[0]:
                        # find a non-outlier chain and replace the three matrices in the right place
                        if np.sum(outlier_mask_m[:, outlier_parallel]) > 0:
                            if np.sum(non_outliers_m[:, outlier_parallel]) > 0:
                                replacement_chains = np.random.choice(np.where(non_outliers_m[:, outlier_parallel])[0],
                                                                      size=np.sum(outlier_mask_m[:, outlier_parallel]))
                                self.log("Replaced chains %s with chains %s in parallel %d" % (
                                    np.where(outlier_mask_m[:, outlier_parallel])[0], replacement_chains, outlier_parallel))

                                # self.log(sum_betas_t[np.where(outlier_mask_m[:,outlier_parallel])[0][0],outlier_parallel,:10])
                                # (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num, print_pc=(outlier_parallel, np.where(outlier_mask_m[:,outlier_parallel])[0][0]))
                                # self.log(np.max(R_m))

                                for tensor in [curr_betas_t, curr_postp_t, curr_post_means_t, sum_betas_t,
                                               sum_betas2_t]:
                                    tensor[outlier_mask_m[:, outlier_parallel], outlier_parallel, :] = copy.copy(
                                        tensor[replacement_chains, outlier_parallel, :])

                                # self.log(sum_betas_t[np.where(outlier_mask_m[:,outlier_parallel])[0][0],outlier_parallel,:10])
                                # (B_m, W_m, R_m, avg_W_m, mean_t) = __calculate_R_tensor(sum_betas_t, sum_betas2_t, iteration_num, print_pc=(outlier_parallel, np.where(outlier_mask_m[:,outlier_parallel])[0][0]))
                                # self.log(np.max(R_m))

                            else:
                                self.log("Every chain was an outlier so doing nothing")

                self.log("Iteration %d: max ind=%s; max B=%.3g; max W=%.3g; max R=%.4g; avg R=%.4g; num above=%.4g;" % (
                    iteration_num,
                    (max_index_parallel, max_index_v[max_index_parallel]) if num_parallel > 1 else max_index_v[
                        max_index_parallel], B_m[max_index_parallel, max_index_v[max_index_parallel]],
                    W_m[max_index_parallel, max_index_v[max_index_parallel]],
                    R_m[max_index_parallel, max_index_v[max_index_parallel]], np.mean(mean_thresholded_R_v),
                    np.sum(R_m > r_threshold_burn_in)))

                converged_v = convergence_statistic_v < r_threshold_burn_in
                newly_converged_v = np.logical_and(burn_in_phase_v, converged_v)
                if np.sum(newly_converged_v) > 0:
                    if num_parallel == 1:
                        self.log("Converged after %d iterations" % iteration_num)
                    else:
                        self.log("Parallel %s converged after %d iterations" % (
                            ",".join([str(p) for p in np.nditer(np.where(newly_converged_v))]), iteration_num))
                    burn_in_phase_v = np.logical_and(burn_in_phase_v, np.logical_not(converged_v))

            if sum(burn_in_phase_v) == 0 or iteration_num >= max_num_burn_in:

                if return_sample:

                    frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
                    if frac_increase > 0.01:
                        self.log(
                            "A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
                        printed_warning_increase = True

                    frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
                    if frac_opposite > 0.01:
                        self.log(
                            "A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
                        printed_warning_swing = False

                    if sum(burn_in_phase_v) > 0:
                        burn_in_phase_v[:] = False
                        self.log("Stopping burn in after %d iterations" % (iteration_num))

                    # max_beta = None
                    # if max_beta is not None:
                    #    threshold_ravel = max_beta * scale_factors_m.ravel()
                    #    if np.sum(sample_betas_m.ravel() > threshold_ravel) > 0:
                    #        self.log("Capped %d sample betas" % np.sum(sample_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        sample_betas_mask = sample_betas_m.ravel() > threshold_ravel
                    #        sample_betas_m.ravel()[sample_betas_mask] = threshold_ravel[sample_betas_mask]
                    #    if np.sum(curr_betas_m.ravel() > threshold_ravel) > 0:
                    #        self.log("Capped %d curr betas" % np.sum(curr_betas_m.ravel() > threshold_ravel), DEBUG)
                    #        curr_betas_mask = curr_betas_m.ravel() > threshold_ravel
                    #        curr_betas_m.ravel()[curr_betas_mask] = threshold_ravel[curr_betas_mask]

                    return (sample_betas_m, sample_postp_m, curr_betas_m, curr_postp_m)

                # average over the posterior means instead of samples
                # these differ from sum_betas_v because those include the burn in phase
                avg_betas_m += np.sum(curr_post_means_t, axis=0)
                avg_betas2_m += np.sum(np.power(curr_post_means_t, 2), axis=0)
                avg_postp_m += np.sum(curr_postp_t, axis=0)
                num_avg += curr_post_means_t.shape[0]

                if iteration_num >= min_num_iter and num_avg > 1:
                    if gauss_seidel:
                        will_break = True
                    else:

                        # calculate these here for trace printing
                        avg_m = avg_betas_m
                        avg2_m = avg_betas2_m
                        sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        sem2_v = np.sum(sem2_m, axis=0)
                        zero_sem2_v = sem2_v == 0
                        sem2_v[zero_sem2_v] = 1
                        total_z_v = np.sqrt(np.sum(avg2_m / num_avg, axis=0) / sem2_v)
                        total_z_v[zero_sem2_v] = np.inf

                        self.log("Iteration %d: sum2=%.4g; sum sem2=%.4g; z=%.3g" % (
                            iteration_num, np.sum(avg2_m / num_avg), np.sum(sem2_m), np.min(total_z_v)))
                        min_z_sampling_var = 10
                        if np.all(total_z_v > min_z_sampling_var):
                            self.log("Desired precision achieved; stopping sampling")
                            will_break = True

                        # TODO: STILL FINALIZING HOW TO DO THIS
                        # avg_m = avg_betas_m
                        # avg2_m = avg_betas2_m

                        # sem2_m = ((avg2_m / (num_avg - 1)) - np.power(avg_m / num_avg, 2)) / num_avg
                        # zero_sem2_m = sem2_m == 0
                        # sem2_m[zero_sem2_m] = 1

                        # max_avg = np.max(np.abs(avg_m / num_avg))
                        # min_avg = np.min(np.abs(avg_m / num_avg))
                        # ref_val = max_avg - min_avg
                        # if ref_val == 0:
                        #    ref_val = np.sqrt(np.var(curr_post_means_t))
                        #    if ref_val == 0:
                        #        ref_val = 1

                        # max_sem = np.max(np.sqrt(sem2_m))
                        # max_percentage_error = max_sem / ref_val

                        # self.log("Iteration %d: ref_val=%.3g; max_sem=%.3g; max_ratio=%.3g" % (iteration_num, ref_val, max_sem, max_percentage_error))
                        # if max_percentage_error < max_frac_sem:
                        #    self.log("Desired precision achieved; stopping sampling")
                        #    break

            else:
                if update_hyper_p or update_hyper_sigma:
                    h2 = 0
                    for i in range(num_parallel):
                        if use_X:
                            h2 += curr_betas_m[i, :].dot(curr_betas_m[i, :])
                        else:
                            if multiple_V:
                                cur_V = V[i, :, :]
                            else:
                                cur_V = V
                            if sparse_V:
                                h2 += V.dot(curr_betas_m[i, :].T).T.dot(curr_betas_m[i, :])
                            else:
                                h2 += curr_betas_m[i, :].dot(cur_V).dot(curr_betas_m[i, :])
                    h2 /= num_parallel

                    new_p = np.mean(
                        (np.sum(curr_betas_t > 0, axis=2) + num_p_pseudo) / float(curr_betas_t.shape[2] + num_p_pseudo))

                    new_sigma2 = h2 / num_gene_sets

                    if num_missing_gene_sets:
                        missing_scale_factor = num_gene_sets / (num_gene_sets + num_missing_gene_sets)
                        new_sigma2 *= missing_scale_factor
                        new_p *= missing_scale_factor

                    if p_noninf_inflate != 1:
                        self.log("Inflating p by %.3g" % p_noninf_inflate)
                        new_p *= p_noninf_inflate

                    if np.all(abs(new_sigma2 - sigma2) / sigma2 < eps) and np.all(abs(new_p - p) / p < eps):
                        self.log("Sigma converged to %.4g; p converged to %.4g" % (sigma2.mean(), p.mean()))
                        update_hyper_sigma = False
                        update_hyper_p = False

            if will_break:
                break

            # self.log("%d\t%s" % (iteration_num, "\t".join(["%.3g\t%.3g" % (curr_betas_m[i,0], (np.mean(sum_betas_m, axis=0) / iteration_num)[i]) for i in range(curr_betas_m.shape[0])])), TRACE)

        avg_betas_m /= num_avg
        avg_postp_m /= num_avg

        if num_parallel == 1:
            avg_betas_m = avg_betas_m.flatten()
            avg_postp_m = avg_postp_m.flatten()

        # max_beta = None
        # if max_beta is not None:
        #    threshold_ravel = max_beta * scale_factors_m.ravel()
        #    if np.sum(avg_betas_m.ravel() > threshold_ravel) > 0:
        #        self.log("Capped %d sample betas" % np.sum(avg_betas_m.ravel() > threshold_ravel), DEBUG)
        #        avg_betas_mask = avg_betas_m.ravel() > threshold_ravel
        #        avg_betas_m.ravel()[avg_betas_mask] = threshold_ravel[avg_betas_mask]

        frac_increase = np.sum(np.abs(curr_betas_m) > np.abs(beta_tildes_m)) / curr_betas_m.size
        if frac_increase > 0.01:
            self.log(
                "A large fraction of betas (%.3g) are larger than beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_increase)
            printed_warning_increase = True

        frac_opposite = np.sum(curr_betas_m * beta_tildes_m < 0) / curr_betas_m.size
        if frac_opposite > 0.01:
            self.log(
                "A large fraction of betas (%.3g) are of opposite signs than the beta tildes; this could indicate a problem. Try increasing --prune-gene-sets value or decreasing --sigma2" % frac_opposite)
            printed_warning_swing = False

        return avg_betas_m, avg_postp_m

    def _finalize_regression(self, beta_tildes, ses, se_inflation_factors):
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
            self.log("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
        p_values = 2 * scipy.stats.norm.cdf(-np.abs(z_scores))
        return beta_tildes, ses, z_scores, p_values, se_inflation_factors

    def compute_beta_tildes(self, X, Y, resid_correlation_matrix=None):
        self.log("Calculating beta tildes")
        y_var = Y.var(axis=1)
        scale_factors = X.std(axis=0)
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

        # multiply by scale factors because we store beta_tilde in units of scaled X
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
            self.log("Adjusting standard errors for correlations")
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

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors)

    def compute_logistic_beta_tildes(self, X, Y, resid_correlation_matrix=None,
                                     convert_to_dichotomous=True, rel_tol=0.01, X_stacked=None, append_pseudo=True):
        self.log("Calculating beta tildes")
        scale_factors = X.std(axis=0)
        # Y can be a matrix with dimensions:
        # number of parallel runs x number of gene sets
        if len(Y.shape) == 1:
            orig_vector = True
            Y = Y[np.newaxis, :]
        else:
            orig_vector = False

        if convert_to_dichotomous:
            if np.sum(np.logical_and(Y != 0, Y != 1)) > 0:
                Y[np.isnan(Y)] = 0
                mult_sum = 1
                # self.log("Multiplying Y sums by %.3g" % mult_sum)
                Y_sums = np.sum(Y, axis=1).astype(int) * mult_sum
                Y_sorted = np.sort(Y, axis=1)[:, ::-1]
                Y_cumsum = np.cumsum(Y_sorted, axis=1)
                threshold_val = np.diag(Y_sorted[:, Y_sums])

                true_mask = (Y.T > threshold_val).T
                Y[true_mask] = 1
                Y[~true_mask] = 0
                self.log("Converting values to dichotomous outcomes; y=1 for input y > %s" % threshold_val)

        # if len(self.genes) == Y.shape[1]:
        #    for i in range(len(self.genes)):
        #        self.log("%s\t%.3g" % (self.genes[i], Y[0,i]))

        self.log("Outcomes: %d=1, %d=0; mean=%.3g" % (np.sum(Y == 1), np.sum(Y == 0), np.mean(Y)))

        len_Y = Y.shape[1]
        num_chains = Y.shape[0]

        if append_pseudo:
            self.log("Appending pseudo counts")
            Y_means = np.mean(Y, axis=1)[:, np.newaxis]

            Y = np.hstack((Y, Y_means))

            X = sparse.csc_matrix(sparse.vstack((X, np.ones(X.shape[1]))))
            if X_stacked is not None:
                X_stacked = sparse.csc_matrix(sparse.vstack((X_stacked, np.ones(X_stacked.shape[1]))))

        # treat multiple chains as just additional gene set coefficients
        if X_stacked is None:
            if num_chains > 1:
                X_stacked = sparse.hstack([X] * num_chains)
            else:
                X_stacked = X

        num_non_zero = np.tile((X != 0).sum(axis=0).A1, num_chains)

        # old, memory more intensive
        # num_non_zero = (X_stacked != 0).sum(axis=0).A1

        num_zero = X_stacked.shape[0] - num_non_zero

        # WHEN YOU LOAD A NON-0/1 GENE SET, USE THIS CODE TO TEST
        # if X.shape[1] > 1:
        #    import statsmodels.api as sm
        #    for i in range(X.shape[1]):
        #        logit_mod = sm.Logit(Y.ravel(), sm.add_constant(X[:,i].todense().A1))
        #        logit_res = logit_mod.fit()
        #        self.log(logit_res.summary())

        # initialize
        # one per gene set
        beta_tildes = np.zeros(X.shape[1] * num_chains)
        # one per gene set
        alpha_tildes = np.zeros(X.shape[1] * num_chains)
        it = 0

        compute_mask = np.full(len(beta_tildes), True)
        diverged_mask = np.full(len(beta_tildes), False)

        def __compute_Y_R(X, beta_tildes, alpha_tildes, max_cap=0.999):
            exp_X_stacked_beta_alpha = X.multiply(beta_tildes)
            exp_X_stacked_beta_alpha.data += (X != 0).multiply(alpha_tildes).data
            max_val = 100
            overflow_mask = exp_X_stacked_beta_alpha.data > max_val
            exp_X_stacked_beta_alpha.data[overflow_mask] = max_val
            np.exp(exp_X_stacked_beta_alpha.data, out=exp_X_stacked_beta_alpha.data)

            # each gene set corresponds to a 2 feature regression
            # Y/R_pred have dim (num_genes, num_chains * num_gene_sets)
            Y_pred = copy.copy(exp_X_stacked_beta_alpha)
            # add in intercepts
            Y_pred.data = Y_pred.data / (1 + Y_pred.data)
            Y_pred.data[Y_pred.data > max_cap] = max_cap
            R = copy.copy(Y_pred)
            R.data = Y_pred.data * (1 - Y_pred.data)
            return (Y_pred, R)

        max_it = 100

        self.log("Performing IRLS...")
        while True:
            it += 1
            prev_beta_tildes = copy.copy(beta_tildes)
            prev_alpha_tildes = copy.copy(alpha_tildes)

            # we are doing num_chains x X.shape[1] IRLS iterations in parallel.
            # Each parallel is a univariate regression of one gene set + intercept
            # first dimension is parallel chains
            # second dimension is each gene set as a univariate regression
            # calculate R
            # X is genesets*chains x genes

            # we are going to do this only for non-zero entries
            # the other entries are technically incorrect, but okay since we are only ever multiplying these by X (which have 0 at these entries)
            (Y_pred, R) = __compute_Y_R(X_stacked[:, compute_mask], beta_tildes[compute_mask], alpha_tildes[compute_mask])

            # values for the genes with zero for the gene set
            # these are constant across all genes (within a gene set/chain)
            max_val = 100
            overflow_mask = alpha_tildes > max_val
            alpha_tildes[overflow_mask] = max_val

            Y_pred_zero = np.exp(alpha_tildes[compute_mask])
            Y_pred_zero = Y_pred_zero / (1 + Y_pred_zero)
            R_zero = Y_pred_zero * (1 - Y_pred_zero)

            Y_sum_per_chain = np.sum(Y, axis=1)
            Y_sum = np.tile(Y_sum_per_chain, X.shape[1])

            # first term: phi*w in Bishop
            # This has length (num_chains * num_gene_sets)

            X_r_X_beta = X_stacked[:, compute_mask].power(2).multiply(R).sum(axis=0).A1.ravel()
            X_r_X_alpha = R.sum(axis=0).A1.ravel() + R_zero * num_zero[compute_mask]
            X_r_X_beta_alpha = X_stacked[:, compute_mask].multiply(R).sum(axis=0).A1.ravel()
            # inverse of [[a b] [c d]] is (1 / (ad - bc)) * [[d -b] [-c a]]
            # a = X_r_X_beta
            # b = c = X_r_X_beta_alpha
            # d = X_r_X_alpha
            denom = X_r_X_beta * X_r_X_alpha - np.square(X_r_X_beta_alpha)

            diverged = np.logical_or(np.logical_or(X_r_X_beta == 0, X_r_X_beta_alpha == 0), denom == 0)

            if np.sum(diverged) > 0:
                self.log("%d beta_tildes diverged" % np.sum(diverged))
                not_diverged = ~diverged

                cur_indices = np.where(compute_mask)[0]

                compute_mask[cur_indices[diverged]] = False
                diverged_mask[cur_indices[diverged]] = True

                # need to convert format in order to support indexing
                Y_pred = sparse.csc_matrix(Y_pred)
                R = sparse.csc_matrix(R)

                Y_pred = Y_pred[:, not_diverged]
                R = R[:, not_diverged]
                Y_pred_zero = Y_pred_zero[not_diverged]
                R_zero = R_zero[not_diverged]
                X_r_X_beta = X_r_X_beta[not_diverged]
                X_r_X_alpha = X_r_X_alpha[not_diverged]
                X_r_X_beta_alpha = X_r_X_beta_alpha[not_diverged]
                denom = denom[not_diverged]

            if np.sum(np.isnan(X_r_X_beta) | np.isnan(X_r_X_alpha) | np.isnan(X_r_X_beta_alpha)) > 0:
                self.log("Error: something went wrong")

            # second term: r_inv * (y-t) in Bishop
            # for us, X.T.dot(Y_pred - Y)

            R_inv_Y_T_beta = X_stacked[:, compute_mask].multiply(Y_pred).sum(axis=0).A1.ravel() - X.T.dot(Y.T).T.ravel()[
                compute_mask]
            R_inv_Y_T_alpha = (Y_pred.sum(axis=0).A1.ravel() + Y_pred_zero * num_zero[compute_mask]) - Y_sum[compute_mask]

            beta_tilde_row = (X_r_X_beta * prev_beta_tildes[compute_mask] + X_r_X_beta_alpha * prev_alpha_tildes[
                compute_mask] - R_inv_Y_T_beta)
            alpha_tilde_row = (X_r_X_alpha * prev_alpha_tildes[compute_mask] + X_r_X_beta_alpha * prev_beta_tildes[
                compute_mask] - R_inv_Y_T_alpha)

            beta_tildes[compute_mask] = (X_r_X_alpha * beta_tilde_row - X_r_X_beta_alpha * alpha_tilde_row) / denom
            alpha_tildes[compute_mask] = (X_r_X_beta * alpha_tilde_row - X_r_X_beta_alpha * beta_tilde_row) / denom

            diff = np.abs(beta_tildes - prev_beta_tildes)
            diff_denom = np.abs(beta_tildes + prev_beta_tildes)
            diff_denom[diff_denom == 0] = 1
            rel_diff = diff / diff_denom

            # self.log("%d left to compute; max diff=%.4g" % (np.sum(compute_mask), np.max(rel_diff)))

            compute_mask[np.logical_or(rel_diff < rel_tol, beta_tildes == 0)] = False
            if np.sum(compute_mask) == 0:
                self.log("Converged after %d iterations" % it)
                break
            if it == max_it:
                self.log("Stopping with %d still not converged" % np.sum(compute_mask))
                diverged_mask[compute_mask] = True
                break

        while True:
            # handle diverged
            if np.sum(diverged_mask) > 0:
                beta_tildes[diverged_mask] = 0
                alpha_tildes[diverged_mask] = Y_sum[diverged_mask] / len_Y

            max_coeff = 100

            # genes x num_coeffs
            (Y_pred, V) = __compute_Y_R(X_stacked, beta_tildes, alpha_tildes)

            # this is supposed to calculate (X^t * V * X)-1
            # where X is the n x 2 matrix of genes x (1/0, 1)
            # d / (ad - bc) is inverse formula
            # a = X.multiply(V).multiply(X)
            # b = c = sum(X.multiply(V)
            # d = V.sum() + constant values for all zero X (since those aren't in V)
            # also need to add in enough p*(1-p) values for all of the X=0 entries; this is where the p_const * number of zero X comes in

            params_too_large_mask = np.logical_or(np.abs(alpha_tildes) > max_coeff, np.abs(beta_tildes) > max_coeff)
            # to prevent overflow
            alpha_tildes[np.abs(alpha_tildes) > max_coeff] = max_coeff

            p_const = np.exp(alpha_tildes) / (1 + np.exp(alpha_tildes))

            # jason_mask = V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1) == 0
            # jason_mask[0] = True
            # if np.sum(jason_mask) > 0:
            #    self.log("JASON ALSO HAVE",np.sum(jason_mask))
            #    self.log(V.sum(axis=0).A1[jason_mask])
            #    self.log((p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))[jason_mask])
            #    self.log(Y_pred.sum(axis=0).A1[jason_mask])
            #    self.log(beta_tildes[jason_mask])
            #    self.log(alpha_tildes[jason_mask])
            #    self.log(p_const[jason_mask])
            #    self.log((X_stacked != 0).sum(axis=0).A1[jason_mask])
            #    self.write_X("x2.gz")
            #    for c in range(Y.shape[0]):
            #        y = ""
            #        for j in range(len(Y[c,:])):
            #            y = "%s\t%s" % (y, Y[c,j])
            #        self.log(y)

            # for i in np.where(jason_mask)[0]:
            #    x = X_stacked[:,i].todense().A1
            #    c = int(i / X.shape[1])
            #    self.log("INDEX",i,"CHAIN",c+1)
            #    for j in range(len(Y[c,:])):
            #        self.log(x[j], Y[c,j])

            variance_denom = (V.sum(axis=0).A1 + p_const * (1 - p_const) * (len_Y - (X_stacked != 0).sum(axis=0).A1))
            denom_zero = variance_denom == 0
            variance_denom[denom_zero] = 1

            variances = X_stacked.power(2).multiply(V).sum(axis=0).A1 - np.power(X_stacked.multiply(V).sum(axis=0).A1,
                                                                                 2) / variance_denom
            variances[denom_zero] = 100

            # set them to diverged also if variances are negative or variance denom is 0 or if params are too large
            additional_diverged_mask = np.logical_and(~diverged_mask,
                                                      np.logical_or(np.logical_or(variances < 0, denom_zero),
                                                                    params_too_large_mask))

            if np.sum(additional_diverged_mask) > 0:
                # additional divergences
                diverged_mask = np.logical_or(diverged_mask, additional_diverged_mask)
            else:
                break

        se_inflation_factors = None
        if resid_correlation_matrix is not None:
            self.log("Adjusting standard errors for correlations")
            # need to multiply by inflation factors: (X * sigma * X) / variances

            if append_pseudo:
                resid_correlation_matrix = sparse.hstack(
                    (resid_correlation_matrix, np.zeros(resid_correlation_matrix.shape[0])[:, np.newaxis]))
                new_bottom_row = np.zeros(resid_correlation_matrix.shape[1])

                new_bottom_row[-1] = 1
                resid_correlation_matrix = sparse.vstack((resid_correlation_matrix, new_bottom_row)).tocsc()

            cor_variances = copy.copy(variances)

            # Old, memory intensive version
            # r_X = resid_correlation_matrix.dot(X_stacked)
            # cor_variances = r_X.multiply(X_stacked).multiply(V).sum(axis=0).A1 - r_X.multiply(V).sum(axis=0).A1 / len_Y

            r_X = resid_correlation_matrix.dot(X)
            # we will only be using this to multiply matrices that are non-zero only when X is
            r_X = (X != 0).multiply(r_X)

            # OLD ONE -- wrong denominator
            # cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack([r_X] * num_chains).multiply(V).sum(axis=0).A1 / len_Y

            # NEW ONE

            cor_variances = sparse.hstack([r_X.multiply(X)] * num_chains).multiply(V).sum(axis=0).A1 - sparse.hstack(
                [r_X] * num_chains).multiply(V).sum(axis=0).A1 / (V.sum(axis=0).A1 + p_const * (1 - p_const) * (
                        len_Y - (X_stacked != 0).sum(axis=0).A1))

            # both cor_variances and variances are in units of unscaled X
            variances[variances == 0] = 1
            se_inflation_factors = np.sqrt(cor_variances / variances)

        # now unpack the chains

        if num_chains > 1:
            beta_tildes = beta_tildes.reshape(num_chains, X.shape[1])
            alpha_tildes = alpha_tildes.reshape(num_chains, X.shape[1])
            variances = variances.reshape(num_chains, X.shape[1])
            diverged_mask = diverged_mask.reshape(num_chains, X.shape[1])
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors.reshape(num_chains, X.shape[1])
        else:
            beta_tildes = beta_tildes[np.newaxis, :]
            alpha_tildes = alpha_tildes[np.newaxis, :]
            variances = variances[np.newaxis, :]
            diverged_mask = diverged_mask[np.newaxis, :]
            if se_inflation_factors is not None:
                se_inflation_factors = se_inflation_factors[np.newaxis, :]

        variances[:, scale_factors == 0] = 1

        # not necessary
        # if inflate_se:
        #    inflate_mask = scale_factors > np.mean(scale_factors)
        #    variances[:,inflate_mask] *= np.mean(np.power(scale_factors, 2)) / np.power(scale_factors[inflate_mask], 2)

        # multiply by scale factors because we store beta_tilde in units of scaled X
        beta_tildes = scale_factors * beta_tildes

        ses = scale_factors / np.sqrt(variances)

        if orig_vector:
            beta_tildes = np.squeeze(beta_tildes, axis=0)
            alpha_tildes = np.squeeze(alpha_tildes, axis=0)
            variances = np.squeeze(variances, axis=0)
            ses = np.squeeze(ses, axis=0)
            diverged_mask = np.squeeze(diverged_mask, axis=0)

            if se_inflation_factors is not None:
                se_inflation_factors = np.squeeze(se_inflation_factors, axis=0)

        return self._finalize_regression(beta_tildes, ses, se_inflation_factors) + (alpha_tildes, diverged_mask)

# TEST =================================================================================================================
# simulationg data
# p_open = 0.13
# n_open = 1000
#
# p_close = 0.0034
# n_close = 10000
#
# a_open = p_open * n_open
# b_open = n_open - a_open
#
# a_close = p_close * n_close
# b_close = n_close - a_close
#
# x_open = np.random.beta(a_open, b_open, n_open)
# x_close = np.random.beta(a_close, b_close, n_close)
#
# data = np.concatenate((np.zeros(n_close), np.ones(n_open)))
# y = np.concatenate((x_close, x_open))
#
# sampled_y = np.random.binomial(1, y).reshape((-1, 1))

# Running regression
# beta_tildes, ses, z_scores, p_values, se_inflation_factors = _compute_beta_tildes(sm.add_constant(data), y.reshape((-1, 1)).T)
# avg_betas_m, avg_postp_m = _calculate_non_inf_betas(beta_tildes[:, 1].reshape((-1, 1)), ses[:, 1].reshape((-1, 1)),
#                                                     data.reshape((-1, 1)), sigma2=np.ones((1, 1))*0.01, p=np.ones((1, 1))*0.1)
# Running logistic regression
# test = SnS()
# beta_tildes, ses, z_scores, p_values, se_inflation_factors, _, _ = \
#     test.compute_logistic_beta_tildes(data.reshape((-1, 1)), sampled_y.reshape((1, -1)))
#
# features_to_keep = test.prune_gene_sets(data.reshape((-1, 1)), p_values, prune_value=0.5)
# avg_betas_m, avg_postp_m = _calculate_non_inf_betas(beta_tildes[:, 1].reshape((-1, 1)), ses[:, 1].reshape((-1, 1)),
#                                                     data.reshape((-1, 1)), sigma2=np.ones((1, 1))*0.01, p=np.ones((1, 1))*0.1)
# self.log('end.')
