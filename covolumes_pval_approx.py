import numpy as np
import scipy.stats as st
from numba import njit, prange

# 1) MGFs and cumulants with zero-based Geom (support x=0,1,2,...)
@njit(fastmath=True)
def compute_K_Kp_Kpp_zero(p_list, q_list, t, h=1e-6):
    """
    Calcola K(t), K'(t), K''(t) per X,Y~Geom0(p),Geom0(q) support {0,1,...}
    con differenze finite e stabilizzazione.
    """
    n = p_list.shape[0]
    max_x = 100  # truncation cutoff for X
    # K(t)
    K = 0.0
    for i in range(n):
        p = p_list[i]
        q = q_list[i]
        s = 0.0
        for x in range(0, max_x+1):
            # P(X=x) = p*(1-p)^x
            px = p * (1.0 - p)**x
            # MGF_Y at t*x: sum_{y=0..inf} q*(1-q)^y e^{t*x*y} = q / (1 - (1-q)*e^{t x})
            ex = np.exp(t * x)
            denom = 1.0 - (1.0 - q) * ex
            if denom <= 0:
                break
            mgf_y = q / denom
            s += px * mgf_y
        if s > 0:
            K += np.log(s)
    # finite difference for derivatives
    K_ph = compute_K_zero(p_list, q_list, t + h, max_x)
    K_mh = compute_K_zero(p_list, q_list, t - h, max_x)
    Kp = (K_ph - K_mh) / (2*h)
    Kpp = (K_ph - 2*K + K_mh) / (h*h)
    return K, Kp, Kpp

@njit(fastmath=True)
def compute_K_zero(p_list, q_list, t, max_x):
    """Helper: compute only K(t) term with zero-based Geom."""
    n = p_list.shape[0]
    K = 0.0
    for i in range(n):
        p = p_list[i]
        q = q_list[i]
        s = 0.0
        for x in range(0, max_x+1):
            px = p * (1.0 - p)**x
            ex = np.exp(t * x)
            denom = 1.0 - (1.0 - q) * ex
            if denom <= 0:
                break
            mgf_y = q / denom
            s += px * mgf_y
        if s > 0:
            K += np.log(s)
    return K

@njit(fastmath=True)
def find_t_star_zero(p_list, q_list, z_obs, tol=1e-6, maxiter=50):
    t = 0.1
    for _ in range(maxiter):
        K, Kp, Kpp = compute_K_Kp_Kpp_zero(p_list, q_list, t)
        diff = Kp - z_obs
        if abs(diff) < tol:
            return t, K, Kpp
        if Kpp == 0:
            break
        t -= diff / Kpp
        if t <= tol:
            t = tol
    K, Kp, Kpp = compute_K_Kp_Kpp_zero(p_list, q_list, t)
    return t, K, Kpp

def saddlepoint_pvalue_zero(p_list, q_list, z_obs):
    p_arr = np.array(p_list, dtype=np.float64)
    q_arr = np.array(q_list, dtype=np.float64)
    t_star, Kt, Kpp_t = find_t_star_zero(p_arr, q_arr, z_obs)
    w = np.sign(t_star) * np.sqrt(max(0.0, 2*(t_star*z_obs - Kt)))
    u = t_star * np.sqrt(max(0.0, Kpp_t))
    return 1 - st.norm.cdf(w) + st.norm.pdf(w)*(1/w - 1/u)

@njit(parallel=True, fastmath=True)
def monte_carlo_zero(p_list, q_list, z_obs, n_samp):
    n = p_list.shape[0]
    cnt = 0
    for i in prange(n_samp):
        total = 0
        for j in range(n):
            # sample Geom0: floor(log(u)/log(1-p))
            u1 = np.random.random()
            x = int(np.log(u1) / np.log(1.0 - p_list[j]))
            u2 = np.random.random()
            y = int(np.log(u2) / np.log(1.0 - q_list[j]))
            total += x * y
        if total >= z_obs:
            cnt += 1
    return cnt / n_samp

def p_value_mix_zero(p_list, q_list, z_obs, mc_samples=10000):
    p_sp = saddlepoint_pvalue_zero(p_list, q_list, z_obs)
    if not np.isfinite(p_sp) or p_sp < 0 or p_sp > 1:
        return monte_carlo_zero(np.array(p_list), np.array(q_list), z_obs, mc_samples)
    return p_sp


