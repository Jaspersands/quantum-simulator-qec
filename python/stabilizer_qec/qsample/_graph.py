"""What qsample's ``Protocol`` uses of networkx's ``DiGraph``, and of simpleeval's
``simple_eval``, without either package."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class _NodeView:
    def __init__(self, graph: "DiGraph") -> None:
        self._g = graph

    def __iter__(self) -> Iterator[Any]:
        return iter(self._g._nodes)

    def __len__(self) -> int:
        return len(self._g._nodes)

    def __contains__(self, n: Any) -> bool:
        return n in self._g._nodes

    def __getitem__(self, n: Any) -> Dict[str, Any]:
        return self._g._nodes[n]

    def __call__(self, data: Any = False, default: Any = None) -> Any:
        if data is False:
            return self
        if data is True:
            return list(self._g._nodes.items())
        return {n: attrs.get(data, default) for n, attrs in self._g._nodes.items()}


class _EdgeView:
    def __init__(self, graph: "DiGraph") -> None:
        self._g = graph

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        return ((u, v) for u, nbrs in self._g._succ.items() for v in nbrs)

    def __len__(self) -> int:
        return sum(len(n) for n in self._g._succ.values())

    def __contains__(self, e: Tuple[Any, Any]) -> bool:
        return e[1] in self._g._succ.get(e[0], {})

    def __getitem__(self, e: Tuple[Any, Any]) -> Dict[str, Any]:
        return self._g._succ[e[0]][e[1]]

    def __call__(self, data: Any = False, default: Any = None) -> Any:
        if data is False:
            return list(self)
        if data is True:
            return [(u, v, a) for u, nbrs in self._g._succ.items() for v, a in nbrs.items()]
        return [(u, v, a.get(data, default)) for u, nbrs in self._g._succ.items() for v, a in nbrs.items()]


class DiGraph:
    """A directed graph with attributes on nodes and edges, as much of networkx's ``DiGraph``
    as qsample's protocols use (``add_node``, ``add_edge``, ``nodes``, ``edges``,
    ``successors``, ``in_degree``, ``out_degree``, ``in_edges``, ``out_edges``, ...)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._nodes: Dict[Any, Dict[str, Any]] = {}
        self._succ: Dict[Any, Dict[Any, Dict[str, Any]]] = {}
        self._pred: Dict[Any, Dict[Any, Dict[str, Any]]] = {}
        self.graph: Dict[str, Any] = dict(kwargs)

    @property
    def nodes(self) -> _NodeView:
        return _NodeView(self)

    @property
    def edges(self) -> _EdgeView:
        return _EdgeView(self)

    def add_node(self, name: Any, **attrs: Any) -> None:
        self._nodes.setdefault(name, {}).update(attrs)
        self._succ.setdefault(name, {})
        self._pred.setdefault(name, {})

    def add_edge(self, u: Any, v: Any, **attrs: Any) -> None:
        for n in (u, v):
            if n not in self._nodes:
                DiGraph.add_node(self, n)
        data = self._succ[u].setdefault(v, {})
        data.update(attrs)
        self._pred[v][u] = data

    def add_edges_from(self, edges: Any, **attrs: Any) -> None:
        for e in edges:
            u, v, *rest = e
            DiGraph.add_edge(self, u, v, **{**attrs, **(rest[0] if rest else {})})

    def has_node(self, n: Any) -> bool:
        return n in self._nodes

    def has_edge(self, u: Any, v: Any) -> bool:
        return v in self._succ.get(u, {})

    def successors(self, n: Any) -> Iterator[Any]:
        return iter(self._succ[n])

    neighbors = successors

    def predecessors(self, n: Any) -> Iterator[Any]:
        return iter(self._pred[n])

    def out_edges(self, n: Any = None) -> List[Tuple[Any, Any]]:
        names = self._nodes if n is None else [n]
        return [(u, v) for u in names for v in self._succ[u]]

    def in_edges(self, n: Any = None) -> List[Tuple[Any, Any]]:
        names = self._nodes if n is None else [n]
        return [(u, v) for v in names for u in self._pred[v]]

    def _degree(self, table: Dict[Any, Dict[Any, Any]], n: Any) -> Any:
        if n is None:
            return [(m, len(table[m])) for m in self._nodes]
        return len(table[n])

    @property
    def in_degree(self) -> "_Degree":
        return _Degree(self, self._pred)

    @property
    def out_degree(self) -> "_Degree":
        return _Degree(self, self._succ)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._nodes)

    def __contains__(self, n: Any) -> bool:
        return n in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def to_networkx(self) -> Any:
        """The same graph as a ``networkx.DiGraph``."""
        import networkx as nx

        g = nx.DiGraph()
        for n, attrs in self._nodes.items():
            g.add_node(n, **attrs)
        for u, nbrs in self._succ.items():
            for v, attrs in nbrs.items():
                g.add_edge(u, v, **attrs)
        return g


class _Degree:
    """``graph.in_degree`` / ``out_degree``: call with a node for its degree, or iterate the
    (node, degree) pairs."""

    def __init__(self, graph: DiGraph, table: Dict[Any, Dict[Any, Any]]) -> None:
        self._g, self._t = graph, table

    def __call__(self, n: Any = None) -> Any:
        return self._g._degree(self._t, n)

    def __getitem__(self, n: Any) -> int:
        return len(self._t[n])

    def __iter__(self) -> Iterator[Tuple[Any, int]]:
        return iter(self._g._degree(self._t, None))


# -- simple_eval -------------------------------------------------------------------------------

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.BitXor: operator.xor, ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_UNARY = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}
_DEFAULT_FUNCTIONS = {"int": int, "float": float, "str": str, "len": len, "abs": abs, "min": min, "max": max, "sum": sum, "any": any, "all": all, "bool": bool}


def simple_eval(expr: str, names: Any = None, functions: Optional[Dict[str, Callable]] = None) -> Any:
    """Evaluates an expression as simpleeval's ``simple_eval`` does: literals, arithmetic,
    comparisons, boolean logic, indexing, conditionals, and calls to ``functions`` only.
    ``names`` is a dict, or a function of the ``ast.Name`` node."""
    funcs = {**_DEFAULT_FUNCTIONS, **(functions or {})}

    def name(node: ast.Name) -> Any:
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        if callable(names):
            return names(node)
        if names is not None and node.id in names:
            return names[node.id]
        raise NameError(f"'{node.id}' is not defined")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return name(node)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                value = True
                for v in node.values:
                    value = ev(v)
                    if not value:
                        return value
                return value
            value = False
            for v in node.values:
                value = ev(v)
                if value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = ev(right_node)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Subscript):
            index = node.slice
            if isinstance(index, ast.Slice):
                key: Any = slice(*(None if x is None else ev(x) for x in (index.lower, index.upper, index.step)))
            else:
                key = ev(index.value if type(index).__name__ == "Index" else index)
            return ev(node.value)[key]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [ev(e) for e in node.elts]
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs and not node.keywords:
            return funcs[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"not allowed in a check: {ast.dump(node)[:80]}")

    return ev(ast.parse(expr.strip(), mode="eval"))
