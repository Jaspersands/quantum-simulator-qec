# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for stabilizer_qec: imports point at this package (``import stabilizer_qec as stim``),
# ``gari`` (which needs scipy) is imported when first used, and the functions here take Stim's
# models as well as this package's, returning the kind they were given.

"""Utilities exported through the public ``tesseract_decoder.demutil`` facade."""

import functools as _functools

from .. import common as _common
from . import decompose_errors as _decompose
from . import detector_basis as _basis
from . import generalize_dem as _generalize
from .detector_basis import (
    automatic_detector_basis_classifier,
    chromobius_detector_basis_classifier,
    last_coordinate_component_classifier,
    stim_surface_code_detector_basis_classifier,
)


def _ours(dem):
    return _common._dem(dem) if _common._is_stims(dem) else dem


def _given_kind(fn, name="dem"):
    """``fn`` taking Stim's model for its ``name`` argument (read as this package's) and
    returning a model of the kind it was given."""

    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args:
            given, args = args[0], (_ours(args[0]),) + args[1:]
        else:
            given = kwargs.get(name)
            if name in kwargs:
                kwargs[name] = _ours(given)
        out = fn(*args, **kwargs)
        return _common._as_given(out, given) if isinstance(out, _common._sq.DetectorErrorModel) else out

    return wrapper


decompose_errors = _given_kind(_decompose.decompose_errors)
decompose_errors_using_detector_basis_classifier = _given_kind(_decompose.decompose_errors_using_detector_basis_classifier)
annotate_detector_bases = _given_kind(_basis.annotate_detector_bases)
classify_detector_bases = _given_kind(_basis.classify_detector_bases)


@_functools.wraps(_generalize.generalize)
def regeneralize_spatial_dem(templates, scaffold, verbose=False):
    out = _generalize.generalize([_ours(t) for t in templates], _ours(scaffold), verbose=verbose)
    return _common._as_given(out, scaffold)

__all__ = [
    "annotate_detector_bases",
    "automatic_detector_basis_classifier",
    "chromobius_detector_basis_classifier",
    "classify_detector_bases",
    "decompose_errors",
    "decompose_errors_using_detector_basis_classifier",
    "gari",
    "last_coordinate_component_classifier",
    "regeneralize_spatial_dem",
    "stim_surface_code_detector_basis_classifier",
]


def __getattr__(name):
    # GARI needs scipy; it is imported when first used, so the rest of demutil does not.
    if name == "gari":
        import importlib

        return importlib.import_module(__name__ + ".gari")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
