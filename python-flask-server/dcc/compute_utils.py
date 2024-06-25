
# imports



# constants



# methods
def _compute_beta_tildes(self, X, Y, y_var, scale_factors, mean_shifts, resid_correlation_matrix=None, log_fun=log):

    log_fun("Calculating beta tildes")

    #Y can be a matrix with dimensions:
    #number of parallel runs x number of gene sets
    if len(Y.shape) == 2:
        len_Y = Y.shape[1]
        Y = (Y.T - np.mean(Y, axis=1)).T
    else:
        len_Y = Y.shape[0]
        Y = Y - np.mean(Y)

    dot_product = np.array(X.T.dot(Y.T) / len_Y).T

    variances = np.power(scale_factors, 2)

    #avoid divide by 0 only
    variances[variances == 0] = 1

    #multiply by scale factors because we store beta_tilde in units of scaled X
    beta_tildes = scale_factors * dot_product / variances

    if len(Y.shape) == 2:
        ses = np.outer(np.sqrt(y_var), scale_factors)
    else:
        ses = np.sqrt(y_var) * scale_factors

    ses /= (np.sqrt(variances * (len_Y - 1)))

    #FIXME: implement exact SEs
    #rather than just using y_var as a constant, calculate X.multiply(beta_tildes)
    #then, subtract out Y for non-zero entries, sum square, sum total
    #then, add in square of Y for zero entries, add in total
    #use these to calculate the variance term

    se_inflation_factors = None
    if resid_correlation_matrix is not None:
        log_fun("Adjusting standard errors for correlations", DEBUG)
        #need to multiply by inflation factors: (X * sigma * X) / variances

        #SEs and betas are stored in units of centered and scaled X
        #we do not need to scale X here, however, because cor_variances will then be in units of unscaled X
        #since variances are also in units of unscaled X, these will cancel out

        r_X = resid_correlation_matrix.dot(X)
        r_X_col_means = r_X.multiply(X).sum(axis=0).A1 / X.shape[0]
        cor_variances = r_X_col_means - np.square(r_X_col_means)
        
        #never increase significance
        cor_variances[cor_variances < variances] = variances[cor_variances < variances]

        #both cor_variances and variances are in units of unscaled X
        se_inflation_factors = np.sqrt(cor_variances / variances)

    return finalize_regression(beta_tildes, ses, se_inflation_factors)

def finalize_regression(beta_tildes, ses, se_inflation_factors):

    if se_inflation_factors is not None:
        ses *= se_inflation_factors

    if np.prod(ses.shape) > 0:
        #empty mask
        empty_mask = np.logical_and(beta_tildes == 0, ses <= 0)
        max_se = np.max(ses)

        ses[empty_mask] = max_se * 100 if max_se > 0 else 100

        #if no y var, set beta tilde to 0

        beta_tildes[ses <= 0] = 0

    z_scores = np.zeros(beta_tildes.shape)
    ses_positive_mask = ses > 0
    z_scores[ses_positive_mask] = beta_tildes[ses_positive_mask] / ses[ses_positive_mask]
    if np.any(~ses_positive_mask):
        warn("There were %d gene sets with negative ses; setting z-scores to 0" % (np.sum(~ses_positive_mask)))
    p_values = 2*scipy.stats.norm.cdf(-np.abs(z_scores))
    return (beta_tildes, ses, z_scores, p_values, se_inflation_factors)


# main



