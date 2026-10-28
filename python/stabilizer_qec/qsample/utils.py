# Ported from qsample 0.0.2 (Copyright Don Winter, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: this package's tableau simulator, and no networkx, anytree, simpleeval or scipy; dill only
# when it is installed.
from __future__ import annotations

__all__ = ['save', 'load']

import os
import pickle
import secrets


def _pickler():
    """dill when it is installed, as qsample saves with it: it also pickles lambdas and nested
    functions, the usual check functions. Without it, the standard library's pickle, which pickles
    only functions defined at a module's top level."""
    try:
        import dill
    except ImportError:
        return pickle
    return dill


def save(data, path):
    """Save data to path

    With dill installed (``pip install dill``), as qsample: a sampler whose protocol has lambda or
    nested check functions saves too. Without it, the standard library's pickle refuses those (it
    pickles only functions defined at a module's top level). Either way the data goes to a new file
    beside ``path`` that replaces it only once written whole, so a save that fails leaves the file
    at ``path`` as it was (qsample opens ``path`` first, emptying it).

    Parameters
    ----------
    path : str
        File path to save to
    data : *
        Data to write to path
    """
    pickler = _pickler()
    path = os.fspath(path)
    tmp = f"{path}.{secrets.token_hex(6)}.tmp"
    try:
        with open(tmp, 'xb') as fp:
            pickler.dump(data, fp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

def load(path):
    """Load data from path (with dill when it is installed, which reads what pickle saves too)

    Parameters
    ----------
    path : str
        File path to load from
        
    Returns
    -------
    *
        Loaded data
    """
    with open(path, 'rb') as fp:  # qsample opens it 'wb', truncating it
        data = _pickler().load(fp)
    return data


def _optional(name, feature):
    """Import ``name``, one of qsample's dependencies that this package does not install (qsample
    requires latextools, matplotlib, networkx and pydot, and uses them only to draw and plot), or
    raise an ImportError naming it and how to install it."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as e:
        package = name.split(".")[0]
        msg = f"{feature} requires the optional dependency '{package}'. Install with: pip install {package}"
        raise ImportError(msg) from e


def tqdm(iterable, **kwargs):
    """tqdm's progress bar when tqdm is installed (as qsample shows one), else the iterable."""
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        return iterable
    return _tqdm(iterable, **kwargs)
