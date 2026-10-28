"""stabilizer_qec.deltakit_stim's samplers of a circuit with leakage (Deltakit's frame simulator's
model, one shot at a time) take any number of observables, as deltakit_stim 0.2.7's do; so does
this package's own leakage sampler."""

import numpy as np

import stabilizer_qec as sq
import stabilizer_qec.deltakit_stim as stim


OBS70 = "R 0\nLEAKAGE(0.5) 0\nHERALD_LEAKAGE_EVENT 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n" + "".join(f"OBSERVABLE_INCLUDE({k}) rec[-2]\n" for k in range(70))


def test_leaky_samplers_take_any_number_of_observables():
    c = stim.Circuit(OBS70)
    assert c.num_observables == 70
    dets, obs = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    assert obs.shape == (2000, 70)
    # Every observable reads the herald, as its detector does.
    assert (obs == dets[:, [1]]).all()
    assert 0.45 < obs[:, 69].mean() < 0.55
    packed = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True, bit_packed=True)[1]
    assert packed.shape == (2000, 9) and (np.unpackbits(packed, axis=1, count=70, bitorder="little") == obs).all()
    m = c.compile_sampler(seed=1).sample(2000)
    assert m.shape == (2000, 2) and 0.45 < m[:, 0].mean() < 0.55
    # This package's own leakage sampler (LEAK) samples them too.
    own = sq.Circuit("R 0\nI_ERROR[LEAK(p=0.5)] 0\nM 0\nDETECTOR rec[-1]\n" + "".join(f"OBSERVABLE_INCLUDE({k}) rec[-1]\n" for k in range(70)))
    d2, o2, _ = own.compile_leakage_sampler(seed=1).sample(2000)
    assert o2.shape == (2000, 70) and (o2 == d2[:, [0]]).all() and 0.45 < o2[:, 69].mean() < 0.55
