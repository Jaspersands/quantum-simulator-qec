"""What qsample's tree uses of anytree: ``Node`` (``name``, ``parent``, ``children``, ``path``,
``iter_path_reverse``, ``leaves``, ``siblings``, ``is_root``, ``is_leaf``, ``root``),
``PreOrderIter`` and ``RenderTree``."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple


class NodeMixin:
    _parent: Optional["NodeMixin"] = None

    @property
    def parent(self) -> Optional["NodeMixin"]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional["NodeMixin"]) -> None:
        old = self._parent
        if old is value:
            return
        if old is not None:
            old._children = [c for c in old._children if c is not self]
        self._parent = value
        if value is not None:
            if "_children" not in value.__dict__:
                value._children = []
            value._children.append(self)

    @property
    def children(self) -> Tuple["NodeMixin", ...]:
        return tuple(self.__dict__.get("_children", ()))

    @children.setter
    def children(self, values: Any) -> None:
        for c in self.children:
            c._parent = None
        self._children = []
        for c in values:
            c.parent = self

    @property
    def path(self) -> Tuple["NodeMixin", ...]:
        return tuple(reversed(list(self.iter_path_reverse())))

    def iter_path_reverse(self) -> Iterator["NodeMixin"]:
        node: Optional[NodeMixin] = self
        while node is not None:
            yield node
            node = node.parent

    @property
    def root(self) -> "NodeMixin":
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def siblings(self) -> Tuple["NodeMixin", ...]:
        if self.parent is None:
            return ()
        return tuple(c for c in self.parent.children if c is not self)

    @property
    def leaves(self) -> Tuple["NodeMixin", ...]:
        return tuple(n for n in PreOrderIter(self) if n.is_leaf)

    @property
    def descendants(self) -> Tuple["NodeMixin", ...]:
        return tuple(PreOrderIter(self))[1:]

    @property
    def depth(self) -> int:
        return len(self.path) - 1


class Node(NodeMixin):
    def __init__(self, name: Any, parent: Optional[NodeMixin] = None, children: Any = None, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.name = name
        self._children = []
        self.parent = parent
        if children:
            self.children = children

    def __repr__(self) -> str:
        return f"Node({'/'.join(str(n.name) for n in self.path)!r})"


def PreOrderIter(node: NodeMixin) -> Iterator[NodeMixin]:
    stack: List[NodeMixin] = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def RenderTree(node: NodeMixin) -> Iterator[Tuple[str, str, NodeMixin]]:
    """(prefix, fill, node) per node, as anytree's ``RenderTree`` yields them."""

    def walk(n: NodeMixin, prefix: str, fill: str, last: bool, top: bool) -> Iterator[Tuple[str, str, NodeMixin]]:
        yield ("" if top else prefix + ("└── " if last else "├── ")), fill, n
        kids = n.children
        for i, c in enumerate(kids):
            yield from walk(c, "" if top else prefix + ("    " if last else "│   "), fill, i == len(kids) - 1, False)

    yield from walk(node, "", "", True, True)
