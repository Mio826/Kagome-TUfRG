"""Finite-temperature Matsubara trace for the restricted three-M GL quartic."""
import numpy as np

def _trace_xy(x, y):
    return np.einsum('kij,kji->k', x, y, optimize=True)

def quartic_loop(H, W, T, mu, nmatsubara=256):
    """Return 3x3: diagonal self Z1; cyclic off-diagonal mixed Z2.

    H: (Nk,24,24), W: (Nk,3,24,24); output per 2x2 supercell.
    Uses positive Matsubara frequencies and their conjugate partners.
    """
    energies, eigvecs = np.linalg.eigh(H)
    rotated = np.einsum('kij,kajl->kail',
                        eigvecs.conj().swapaxes(1,2), W, optimize=True)
    rotated = np.einsum('kail,klm->kaim', rotated, eigvecs, optimize=True)
    coeffs = np.zeros((3,3),float)
    for m in range(nmatsubara):
        wn = (2*m+1)*np.pi*T
        d = 1/(1j*wn-(energies-mu))
        A = [d[:,:,None]*rotated[:,i] for i in range(3)]
        sq = [x@x for x in A]
        for i in range(3):
            coeffs[i,i] += T*_trace_xy(sq[i],sq[i]).mean().real
            j = (i+1)%3
            aa = _trace_xy(sq[i],sq[j]).mean()
            bb = _trace_xy(A[i]@A[j],A[i]@A[j]).mean()
            coeffs[i,j] += 2*T*(aa+0.5*bb).real
    return coeffs
