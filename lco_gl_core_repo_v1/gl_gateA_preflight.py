"""Source-backed folded 2x2 kagome H0 construction."""
from __future__ import annotations
import numpy as np
import TUfRG_model as model
CELLS=[(0,0),(1,0),(0,1),(1,1)]
CELL_I={c:i for i,c in enumerate(CELLS)}


def folded_h0(K, hops):
    """24x24 folded H0; K in supercell reciprocal reduced coordinates.

    The source hopping matrix element H_{o1,o2}(R) is stored by the
    reference model with Bloch factor +2pi*i*k.R (same as Hk_reduced).
    """
    K = np.asarray(K,dtype=float)
    assert K.shape == (2,)
    H = np.zeros((24,24), dtype=complex)
    for cell_i, cell in enumerate(CELLS):
        for R,o1,o2,s1,s2,(real,imag) in hops:
            endpoint = (cell[0]+int(R[0]), cell[1]+int(R[1]))
            destcell = (endpoint[0]%2,endpoint[1]%2)
            wrap = ((endpoint[0]-destcell[0])//2,(endpoint[1]-destcell[1])//2)
            row = int(s1)*12+cell_i*3+int(o1)
            col = int(s2)*12+CELL_I[destcell]*3+int(o2)
            H[row,col] += (real+1j*imag)*np.exp(2j*np.pi*np.dot(K,wrap))
    return H


def primitive_folded_spectrum(K,hops):
    """Independent 4*k sectors from the original primitive H(k)."""
    eigs=[]
    for m1 in (0,1):
        for m2 in (0,1):
            k=np.array([(K[0]+m1)/2, (K[1]+m2)/2, 0.])
            eigs.extend(np.linalg.eigvalsh(model.Hk_reduced(k,hops)))
    return np.sort(np.asarray(eigs))
