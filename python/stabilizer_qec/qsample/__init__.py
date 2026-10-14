"""qsample (Winter et al.'s dynamical subset sampling of fault-tolerant protocols), on this
package's stabilizer simulator.

Protocols are graphs of circuits whose transitions check the measurement history
(``Protocol``, ``Circuit`` in qsample's tick format), noise is placed by error models
(``E1``, ``E2``, ``E3``, ``S1``, ...), and ``DirectSampler`` samples them directly while
``SubsetSampler`` samples each fault-count subset and weighs it by its probability, giving the
logical failure rate at every physical error rate below ``p_max`` from one run, with the
statistical error and the upper bound from subsets never reached. As qsample 0.0.2 (Apache
License 2.0; see THIRD_PARTY_NOTICES.md), without its dependencies beyond numpy."""

from stabilizer_qec.qsample import callbacks, math, noise, utils
from stabilizer_qec.qsample.circuit import Circuit
from stabilizer_qec.qsample.noise import *  # noqa: F401,F403 - qsample's error models
from stabilizer_qec.qsample.protocol import Protocol, draw_protocol
from stabilizer_qec.qsample.sampler.direct import DirectSampler
from stabilizer_qec.qsample.sampler.subset import SubsetSampler
from stabilizer_qec.qsample.sim import StabilizerSimulator, StatevectorSimulator

__version__ = "0.0.2"  # the version of qsample ported
