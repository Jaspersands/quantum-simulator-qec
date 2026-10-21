# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/detectors/database.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec: the database lives in memory only. tqec's reading and writing of
# pickle and JSON files (and its semver version) are not ported; the key, and the lookups and
# insertions the detector computation makes, are tqec's.
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from stabilizer_qec.tqec._compile.detectors.detector import Detector
from stabilizer_qec.tqec._compile.plaquette.plaquette import Plaquettes
from stabilizer_qec.tqec._compile.templates.subtemplates import SubTemplateType
from stabilizer_qec.tqec._errors import TQECError

# tqec's semver.Version(1, 0, 0), as a tuple.
CURRENT_DATABASE_VERSION: Final[tuple[int, int, int]] = (1, 0, 0)


@dataclass(frozen=True)
class _DetectorDatabaseKey:
    """Immutable type used as a key in the database of detectors.

    This class represents a "situation" for which we might be able to compute
    several detectors. Its purpose of existence is to provide sensible
    `__hash__` and `__eq__` operations in order to be able to use a "situation"
    as a `dict` key.

    Attributes:
        subtemplates: a sequence of 2-dimensional arrays of integers representing
            the sub-template(s). Each entry corresponds to one QEC round.
        plaquettes_by_timestep: a list of :class:`Plaquettes`, each
            :class:`Plaquettes` entry storing enough :class:`Plaquette`
            instances to generate a circuit from corresponding entry in
            `self.subtemplates` and corresponding to one QEC round.

    ## Implementation details

    This class uses a surjective representation to compare (`__eq__`) and hash
    (`__hash__`) its instances. This representation is computed and cached using
    the :meth:`_DetectorDatabaseKey.plaquette_names` property that basically
    uses the provided subtemplates to build a nested tuple data-structure with
    the same shape as `self.subtemplates` (3 dimensions, the first one being the
    number of time steps, the next 2 ones being of odd and equal size and
    depending on the radius used to build subtemplates) storing in each of its
    entries the corresponding plaquette name.

    This intermediate data-structure is not the most memory efficient one, but
    it has the advantage of being easy to construct, trivially invariant to
    plaquette re-indexing and easy to hash (with some care to NOT use Python's
    default `hash` due to its absence of stability across different runs).

    """

    subtemplates: Sequence[SubTemplateType]
    plaquettes_by_timestep: Sequence[Plaquettes]

    def __post_init__(self) -> None:
        if len(self.subtemplates) != len(self.plaquettes_by_timestep):
            raise TQECError(
                "DetectorDatabaseKey can only store an equal number of "
                f"subtemplates and plaquettes. Got {len(self.subtemplates)} "
                f"subtemplates and {len(self.plaquettes_by_timestep)} plaquettes."
            )

    @property
    def num_timeslices(self) -> int:
        return len(self.subtemplates)

    @cached_property
    def plaquette_names(self) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Cached property returning a representation of the current situation with plaquette names.

        Returns:
            Nested tuples such that
            ``ret[t][y][x] == self.plaquettes_by_timestep[t][self.subtemplates[t][y, x]].name``.

            The returned object can be iterated on using:

            .. code:: python

                for t, names in enumerate(self.plaquette_names):
                    subtemplate = self.subtemplates[t]
                    plaquettes = self.plaquettes_by_timestep[t]
                    for y, names_row in enumerate(names):
                        for x, name in enumerate(names_row):
                            plaquette: Plaquette = plaquettes[subtemplate[y, x]]
                            assert name == plaquette.name

        """
        return tuple(
            tuple(tuple(plaquettes[pi].name for pi in row) for row in st)
            for st, plaquettes in zip(self.subtemplates, self.plaquettes_by_timestep)
        )

    @cached_property
    def reliable_hash(self) -> int:
        """Return a hash of ``self`` that is guaranteed to be constant.

        Python's ``hash`` is not guaranteed to be constant across Python versions, OSes and
        executions. In particular, strings hash will not be repeatable across different Python
        executable calls. The following line should return different values at each call:

        .. code:: bash

            python -c 'print(hash("Hello world"))'

        For example, here are the results obtained after calling that line 3 times on my machine:

        - ``2656127643635015930``
        - ``1413792191058799258``
        - ``8731178165517315210``

        This is an issue for the detector database as we would like it to be reproducible. Else,
        reusing an existing database will always fail because keys will have a different hash,
        hence re-computing detectors at each call and growing the database indefinitely.

        This method implements a reliable hash that should be constant no matter the context
        (different Python calls, different OS, different version of Python, ...).
        """
        hasher = hashlib.md5()
        for timeslice in self.plaquette_names:
            for row in timeslice:
                for name in row:
                    hasher.update(name.encode())
        return int(hasher.hexdigest(), 16)

    def __hash__(self) -> int:
        return self.reliable_hash

    def __eq__(self, rhs: object) -> bool:
        return isinstance(rhs, _DetectorDatabaseKey) and self.plaquette_names == rhs.plaquette_names


class DetectorDatabase:
    version: tuple[int, int, int] = (0, 0, 0)

    def __init__(
        self,
        mapping: dict[_DetectorDatabaseKey, frozenset[Detector]] | None = None,
        frozen: bool = False,
    ):
        """Store a mapping from "situations" to the corresponding detectors.

        This class aims at storing efficiently a set of "situations" in which the
        corresponding detectors are known and do not have to be re-computed.

        In this class, a "situation" is described by :class:`_DetectorDatabaseKey`
        and correspond to a spatially and temporally local piece of a larger
        computation.

        The version number should be manually updated when code is pushed which makes old
        instances of the database incompatible with newly generated instances.
        Guidance on when to change `a` (major) or `b` (minor) in the `a.b` version number:
        - MAJOR when the format of the file changes (i.e. when the attributes of
        ``DetectorDatabase`` change),
        - MINOR when the content of the database is invalidated (e.g. by changing a plaquette
        implementation without changing its name).

        Old databases generated prior to the introduction of a version attribute will be
        loaded with the default value of .version, without passing through __init__,
        ie (0,0,0).

        """
        if mapping is None:
            mapping = dict()
        self.mapping = mapping
        self.frozen = frozen
        self.version = CURRENT_DATABASE_VERSION

    def add_situation(
        self,
        subtemplates: Sequence[SubTemplateType],
        plaquettes_by_timestep: Sequence[Plaquettes],
        detectors: frozenset[Detector] | Detector,
    ) -> None:
        """Add a new situation to the database.

        Args:
            subtemplates: a sequence of 2-dimensional arrays of integers
                representing the sub-template(s). Each entry corresponds to one
                QEC round.
            plaquettes_by_timestep: a list of :class:`Plaquettes`, each
                :class:`Plaquettes` entry storing enough :class:`Plaquette`
                instances to generate a circuit from corresponding entry in
                `self.subtemplates` and corresponding to one QEC round.
            detectors: computed detectors that should be stored in the database.
                The coordinates used by the :class:`Measurement` instances stored
                in each entry should be relative to the top-left qubit of the
                top-left plaquette in the provided `subtemplates`.

        Raises:
            TQECError: if this method is called and `self.frozen`.

        """
        if self.frozen:
            raise TQECError("Cannot add a situation to a frozen database.")
        key = _DetectorDatabaseKey(subtemplates, plaquettes_by_timestep)
        self.mapping[key] = frozenset([detectors]) if isinstance(detectors, Detector) else detectors

    def remove_situation(
        self,
        subtemplates: Sequence[SubTemplateType],
        plaquettes_by_timestep: Sequence[Plaquettes],
    ) -> None:
        """Remove an existing situation from the database.

        Args:
            subtemplates: a sequence of 2-dimensional arrays of integers
                representing the sub-template(s). Each entry corresponds to one
                QEC round.
            plaquettes_by_timestep: a list of :class:`Plaquettes`, each
                :class:`Plaquettes` entry storing enough :class:`Plaquette`
                instances to generate a circuit from corresponding entry in
                `self.subtemplates` and corresponding to one QEC round.

        Raises:
            TQECError: if this method is called and `self.frozen`.

        """
        if self.frozen:
            raise TQECError("Cannot remove a situation to a frozen database.")
        key = _DetectorDatabaseKey(subtemplates, plaquettes_by_timestep)
        del self.mapping[key]

    def get_detectors(
        self,
        subtemplates: Sequence[SubTemplateType],
        plaquettes_by_timestep: Sequence[Plaquettes],
    ) -> frozenset[Detector] | None:
        """Return the detectors associated with the provided situation.

        Args:
            subtemplates: a sequence of 2-dimensional arrays of integers
                representing the sub-template(s). Each entry corresponds to one
                QEC round.
            plaquettes_by_timestep: a list of :class:`Plaquettes`, each
                :class:`Plaquettes` entry storing enough :class:`Plaquette`
                instances to generate a circuit from corresponding entry in
                `self.subtemplates` and corresponding to one QEC round.

        Returns:
            detectors associated with the provided situation or `None` if the
            situation is not in the database.

        """
        key = _DetectorDatabaseKey(subtemplates, plaquettes_by_timestep)
        return self.mapping.get(key)

    def freeze(self) -> None:
        """Make ``self`` read-only."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Make ``self`` writable."""
        self.frozen = False

    def __len__(self) -> int:
        return len(self.mapping)
