"""The two immutable containers MWPF's Python helpers import from other packages, written here so
they need neither: ``frozendict`` (the ``frozendict`` package's, a hashable read-only dict) and
``FrozenList`` (``frozenlist``'s, a list that refuses changes once ``freeze()`` is called, and is
hashable only then)."""

import types
from collections.abc import MutableSequence

__all__ = ["frozendict", "FrozenList"]


def _immutable(self, *args, **kwargs):
    raise TypeError(f"'{type(self).__name__}' object is immutable")


def _read_only(self, *args, **kwargs):
    raise AttributeError(f"'{type(self).__name__}' object is read-only")


class frozendict(dict):
    """A dict that cannot change after it is made, and so can be hashed; its changes (``set``,
    ``delete``, ``setdefault``) return new ones, as frozendict 2.4's do."""

    __slots__ = ("_hash",)

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self.items()))
        return self._hash

    def __repr__(self):
        return f"frozendict.frozendict({dict.__repr__(self)})"

    __str__ = __repr__

    def __reduce__(self):
        return (type(self), (dict(self),))

    def copy(self):
        return self

    def __or__(self, other):
        return type(self)({**self, **other})

    def set(self, key, value):
        return type(self)({**self, key: value})

    def delete(self, key):
        if key not in self:
            raise KeyError(key)
        return type(self)({k: v for k, v in self.items() if k != key})

    def setdefault(self, key, default=None):
        return self if key in self else self.set(key, default)

    def _at(self, index):
        items = tuple(dict.items(self))
        if isinstance(index, int) and not -len(items) <= index < len(items):
            raise IndexError(f"{type(self).__name__} index {index} out of range {len(items) - 1}")
        return items[index]

    def key(self, index=0):
        return self._at(index)[0]

    def value(self, index=0):
        return self._at(index)[1]

    def item(self, index=0):
        return self._at(index)

    def __setitem__(self, key, value):
        raise TypeError(f"'{type(self).__name__}' object doesn't support item assignment")

    def __delitem__(self, key):
        raise TypeError(f"'{type(self).__name__}' object doesn't support item deletion")

    clear = pop = popitem = update = __ior__ = _read_only


class FrozenList(MutableSequence):
    """A list that can be frozen: after ``freeze()`` it refuses changes and becomes hashable."""

    __slots__ = ("_frozen", "_items")
    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, items=None):
        self._frozen = False
        self._items = list(items) if items is not None else []

    @property
    def frozen(self):
        return self._frozen

    def freeze(self):
        self._frozen = True

    def _check(self):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen list.")

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._check()
        self._items[index] = value

    def __delitem__(self, index):
        self._check()
        del self._items[index]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __eq__(self, other):
        return list(self) == other

    def __lt__(self, other):
        return list(self) < other

    def __le__(self, other):
        return list(self) <= other

    def __gt__(self, other):
        return list(self) > other

    def __ge__(self, other):
        return list(self) >= other

    def insert(self, pos, item):
        self._check()
        self._items.insert(pos, item)

    def __repr__(self):
        return f"<FrozenList(frozen={self._frozen}, {self._items!r})>"

    def __hash__(self):
        if self._frozen:
            return hash(tuple(self))
        raise RuntimeError("Cannot hash unfrozen list.")
