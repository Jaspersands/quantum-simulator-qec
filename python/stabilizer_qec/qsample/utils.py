# Ported from qsample 0.0.2 (Copyright Don Winter, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: this package's tableau simulator, and no networkx, anytree, simpleeval, scipy or dill.
from __future__ import annotations

__all__ = ['save', 'load']

import pickle

def save(data, path):
    """Save data to path

    Parameters
    ----------
    path : str
        File path to save to
    data : *
        Data to write to path
    """
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def load(path):
    """Load data from path

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
        data = pickle.load(fp)
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
