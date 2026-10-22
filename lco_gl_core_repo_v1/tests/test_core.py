"""Dependency-free (except numpy) smoke tests for repo-safe GL code."""
import unittest
import numpy as np
import gl_gateA_preflight as pre
from gl_quartic import quartic_loop
from gl_thermo import choose_mu


class CoreTests(unittest.TestCase):
    def test_folded_bloch_spectrum(self):
        hops = pre.model.spinful_hoppings(t=1., mu=0., lambda_nn=.5,
                                         lambda_nnn=0., lambda_rashba=0.)
        for k in [(0.0,0.0),(.19,.37),(.53,.11)]:
            h = pre.folded_h0(k,hops)
            self.assertLess(np.linalg.norm(h-h.conj().T),1e-11)
            np.testing.assert_allclose(np.linalg.eigvalsh(h),
                                       pre.primitive_folded_spectrum(k,hops),atol=1e-11)

    def test_half_filling_mu(self):
        levels = np.repeat(np.array([[-2., 2.]]), 4, axis=0)
        self.assertAlmostEqual(choose_mu(levels,.2,expected_per_sc=1),0.,places=12)

    def test_quartic_trace_scalar_algebra(self):
        # Deliberately commuting matrices: coefficient ratio is exact.
        temp = .35
        nfreq = 8
        h = np.zeros((2,2,2),dtype=complex)
        w = np.zeros((2,3,2,2),dtype=complex)
        amp = [1.,2.,.5]
        for a in range(3):
            w[:,a] = amp[a]*np.eye(2)
        z = quartic_loop(h,w,temp,0.,nfreq)
        common = 2*temp*sum(1/((2*m+1)*np.pi*temp)**4 for m in range(nfreq))
        for a in range(3):
            self.assertAlmostEqual(z[a,a], common*amp[a]**4,places=10)
            b = (a+1)%3
            self.assertAlmostEqual(z[a,b],3*common*amp[a]**2*amp[b]**2,places=10)

    def test_benchmark_if_configured(self):
        # Real FRG data are deliberately NOT committed to the code repository.
        import os
        archive=os.environ.get('LCO_GL_REFERENCE_ARCHIVE')
        fmap=os.environ.get('LCO_GL_REFERENCE_MAP')
        if not (archive and fmap):
            self.skipTest('set LCO_GL_REFERENCE_ARCHIVE and LCO_GL_REFERENCE_MAP')
        from gl_gateB_reusable import compute
        r=compute(fmap,archive,'iter_0074',.12,12,128)
        delta=r['quartic_per_2by2_supercell']['Z1_minus_Z2']
        self.assertAlmostEqual(delta,0.118031080987,places=8)

if __name__=='__main__':
    unittest.main()
