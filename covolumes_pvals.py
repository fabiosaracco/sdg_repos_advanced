import numpy as np
from numba import njit, prange

# 1) JIT-compilazione della pmf del prodotto geometrico
@njit(fastmath=True)
def pmf_geom_product_jit(p, q, M):
    """
    Restituisce f[0..M] con f[k] = Pr(X*Y = k)
    per X~Geom(p), Y~Geom(q), indipendenti.
    Il supporto inizia da k=1, f[0]=0.
    """
    f = np.zeros(M+1, dtype=np.float64)
    pq = p*q
    for d in range(1, M+1):
        # (1-p)^(d-1)
        pd = (1.0 - p)**(d-1)
        coef = pq * pd
        term = 1.0  # (1-q)^(m-1)
        max_m = M // d
        for m in range(1, max_m+1):
            km = d * m
            f[km] += coef * term
            term *= (1.0 - q)
    return f

def p_value_dot_product_numba(p_list, q_list, z_obs, M=None):
    """
    Calcola esattamente Pr(Z >= z_obs), Z = sum_i X_i Y_i,
    X_i~Geom(p_i), Y_i~Geom(q_i), indipendenti,
    usando:
      - pmf_geom_product_jit() per generare ciascuna f_i in O(M)
      - una sola FFT finale sul prodotto delle FFT di f_i
    Complessità: O(n·M + M log M).
    """
    n = len(p_list)
    if M is None:
        M = z_obs

    # dimensione FFT per evitare aliasing lineare: L >= 2M+1
    L = 2*M + 1

    # accumula il prodotto delle FFT reali
    freq_prod = np.ones(L//2 + 1, dtype=np.complex128)

    # 2) per ogni coppia (p_i, q_i) genera f_i e ne moltiplica la FFT
    for p, q in zip(p_list, q_list):
        f_i = pmf_geom_product_jit(p, q, M)
        fft_fi = np.fft.rfft(f_i, n=L)
        freq_prod *= fft_fi

    # 3) un’unica IFFT per ottenere la pmf totale
    conv = np.fft.irfft(freq_prod, n=L)
    total_pmf = conv[:M+1].real
    total_pmf /= total_pmf.sum()  # corregge eventuali drift numerici

    # 4) p-value come coda
    return float(total_pmf[z_obs:].sum())