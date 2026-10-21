# Ported from tqec (tqec/tqec 0.2.0, src/tqec/{circuit,plaquette,templates,compile}/ and the utils
# modules they use; Apache License 2.0, see LICENSE in stabilizer_qec/tqec) and tqecd (tqec/tqecd
# 0.2.1, in tqecd/; Apache License 2.0, see tqecd/LICENSE), modified for stabilizer_qec.
"""tqec's compile pipeline: block graphs to plaquettes, templates and layers, and those to Stim
circuits with tqecd's detectors and tqec's observables, as tqec 0.2.0 writes them, byte for
byte. The public entry is ``stabilizer_qec.tqec.compile_block_graph``. This package keeps tqec's
module structure, with ``tqec.compile``'s modules (``blocks``, ``specs``, ``detectors``,
``observables``, ``tree``, ``graph``, ``generation``, ``convention``) at its top level beside
``circuit``, ``plaquette``, ``templates`` and ``utils``, so each module reads against its
original."""
