"""Normal-state chemical potential for a requested supercell filling."""
import numpy as np

def choose_mu(eigs, T, expected_per_sc=10.0):
    if T <= 0:
        raise ValueError("Requires T > 0")
    lo = float(np.min(eigs) - 20*T - 1)
    hi = float(np.max(eigs) + 20*T + 1)
    for _ in range(90):
        mid = (lo+hi)/2
        n = np.mean(np.sum(np.exp(-np.logaddexp(0., (eigs-mid)/T)), axis=1))
        if n > expected_per_sc:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2
