import numpy as np
from scipy.stats import norm

def calculate_cq_statistic(X, Y):
    """
    Computes the Chen-Qin (2010) U-statistic for the equality of two high-dimensional means.
    This serves as an unbiased estimator of ||mu_x - mu_y||^2 without cross-product bias.
    """
    n, p = X.shape
    m, _ = Y.shape
    
    # Sum of vectors
    sum_X = X.sum(axis=0)
    sum_Y = Y.sum(axis=0)
    
    # Sum of squared row norms
    sq_norm_X = np.sum(X**2)
    sq_norm_Y = np.sum(Y**2)
    
    # Component 1: Unbiased term for X
    term_X = (np.dot(sum_X, sum_X) - sq_norm_X) / (n * (n - 1))
    
    # Component 2: Unbiased term for Y
    term_Y = (np.dot(sum_Y, sum_Y) - sq_norm_Y) / (m * (m - 1))
    
    # Component 3: Interaction term
    term_XY = (2.0 / (n * m)) * np.dot(sum_X, sum_Y)
    
    # Chen-Qin statistic
    t_cq = term_X + term_Y - term_XY
    return t_cq

def calculate_bs_statistic(X, Y):
    """
    Computes the centralized Bai-Saranadasa (1996) L2-norm statistic.
    Assumes homoscedasticity (equal covariance structures).
    """
    n, p = X.shape
    m, _ = Y.shape
    
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    
    # Pooled sample covariance trace
    pooled_cov = ((n - 1) * np.cov(X, rowvar=False) + (m - 1) * np.cov(Y, rowvar=False)) / (n + m - 2)
    tr_cov = np.trace(pooled_cov)
    
    # Squared Euclidean distance of means
    mean_diff_sq = np.sum((mean_X - mean_Y) ** 2)
    
    # Bai-Saranadasa statistic
    t_bs = (n * m / (n + m)) * mean_diff_sq - tr_cov
    return t_bs

def high_dim_two_sample_test(X, Y, method='chen-qin', num_permutations=1000, random_state=42):
    """
    Executes a high-dimensional multivariate mean test using Permutations.
    """
    np.random.seed(random_state)
    n = X.shape[0]
    combined = np.vstack([X, Y])
    
    # Select test statistic logic
    if method.lower() == 'chen-qin' or method.lower() == 'cq':
        stat_func = calculate_cq_statistic
        method_name = "Chen-Qin (2010) High-Dimensional Test"
    elif method.lower() == 'bai-saranadasa' or method.lower() == 'bs':
        stat_func = calculate_bs_statistic
        method_name = "Bai-Saranadasa (1996) High-Dimensional Test"
    else:
        raise ValueError("Method must be 'chen-qin' or 'bai-saranadasa'")
        
    # Calculate observed statistic
    observed_stat = stat_func(X, Y)
    
    # Permutation loop
    count = 0
    for _ in range(num_permutations):
        permuted_data = np.random.permutation(combined)
        perm_X = permuted_data[:n]
        perm_Y = permuted_data[n:]
        
        if stat_func(perm_X, perm_Y) >= observed_stat:
            count += 1
            
    p_value = count / num_permutations
    
    return {
        "method": method_name,
        "statistic": observed_stat,
        "p_value": p_value
    }
