import numpy as np
from scipy.stats import norm

def high_dim_analytical_tests(X, Y):
    """
    Computes analytical p-values for both the Bai-Saranadasa (1996) 
    and Chen-Qin (2010) high-dimensional mean tests.
    """
    n, p = X.shape
    m, _ = Y.shape
    
    # ----------------------------------------------------
    # Precomputations & Shared Variables
    # ----------------------------------------------------
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    
    # Sample Covariances
    S_X = np.cov(X, rowvar=False)
    S_Y = np.cov(Y, rowvar=False)
    
    # Unbiased estimators for Tr(Sigma^2) components
    # (Required to build the analytical variances)
    def estimate_tr_sigma2(S, N):
        # Srivastava's ratio-consistent estimator for tr(Sigma^2)
        tr_S2 = np.trace(S @ S)
        tr_S_sq = np.trace(S) ** 2
        return (N**2 / ((N + 1) * (N - 2))) * (tr_S2 - (tr_S_sq / N))

    tr_Sigma_X2 = estimate_tr_sigma2(S_X, n)
    tr_Sigma_Y2 = estimate_tr_sigma2(S_Y, m)
    
    # Interaction trace estimator: tr(Sigma_X * Sigma_Y)
    tr_Sigma_XY = np.trace(S_X @ S_Y)

    # ----------------------------------------------------
    # 1. CHEN-QIN (2010) TEST (Handles Unequal Covariances)
    # ----------------------------------------------------
    # Raw Statistic Calculation
    sum_X = X.sum(axis=0)
    sum_Y = Y.sum(axis=0)
    sq_norm_X = np.sum(X**2)
    sq_norm_Y = np.sum(Y**2)
    
    term_X = (np.dot(sum_X, sum_X) - sq_norm_X) / (n * (n - 1))
    term_Y = (np.dot(sum_Y, sum_Y) - sq_norm_Y) / (m * (m - 1))
    term_XY = (2.0 / (n * m)) * np.dot(sum_X, sum_Y)
    
    stat_cq = term_X + term_Y - term_XY
    
    # Analytical Variance of the Chen-Qin estimator
    var_cq = (2 / (n * (n - 1))) * tr_Sigma_X2 + \
             (2 / (m * (m - 1))) * tr_Sigma_Y2 + \
             (4 / (n * m)) * tr_Sigma_XY
             
    z_cq = stat_cq / np.sqrt(var_cq)
    p_cq = 1.0 - norm.cdf(z_cq)  # One-tailed upper tail test

    # ----------------------------------------------------
    # 2. BAI-SARANDASA (1996) TEST (Assumes Equal Covariances)
    # ----------------------------------------------------
    # Pooled Covariance Matrix
    S_pooled = ((n - 1) * S_X + (m - 1) * S_Y) / (n + m - 2)
    tr_S_pooled = np.trace(S_pooled)
    
    # Raw Statistic
    mean_diff_sq = np.sum((mean_X - mean_Y) ** 2)
    stat_bs = (n * m / (n + m)) * mean_diff_sq - tr_S_pooled
    
    # Analytical Variance under homoscedasticity 
    tr_Sigma_pooled2 = estimate_tr_sigma2(S_pooled, n + m)
    var_bs = 2 * (1 + (n * m / (n + m)) / (n + m)) * tr_Sigma_pooled2
    
    z_bs = stat_bs / np.sqrt(var_bs)
    p_bs = 1.0 - norm.cdf(z_bs)

    return {
        "Chen-Qin (2010)": {"statistic": stat_cq, "z_score": z_cq, "p_value": p_cq},
        "Bai-Saranadasa (1996)": {"statistic": stat_bs, "z_score": z_bs, "p_value": p_bs}
    }
