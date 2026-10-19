# Writes YAML as LLVM's llvm::yaml::Output does (llvm/lib/Support/YAMLTraits.cpp, LLVM Project,
# Apache License 2.0 with LLVM Exceptions), which is what CUDA-Q QEC's decoder configurations are
# written with; reads the YAML subset those configurations use. Pure Python, no YAML library.
"""YAML for CUDA-Q QEC's real-time decoding configurations.

``parse`` reads the first document of a YAML text into a tree of `Node`: block mappings and
sequences (including a mapping started on a ``- `` line), flow sequences and mappings (also
across lines), plain, single- and double-quoted scalars, comments and the ``---`` / ``...``
markers. Anchors, aliases, tags and block scalars (``|``, ``>``) are refused, not guessed at.

``document`` writes a tree of emit values as LLVM does: keys padded so values start 16 columns
past the key, flow sequences ``[ a, b ]`` wrapped once a line passes ``column_wrap``, scalars quoted
by LLVM's rules (``needs_quotes``), ``---`` before and ``...`` after."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

# -- Reading --------------------------------------------------------------------------------------


class YamlError(Exception):
    """A YAML syntax error at a 1-based line and column."""

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.message, self.line, self.col = message, line, col


class Node:
    """A parsed node: ``kind`` "map" (``value`` a list of (key, key node, value node)), "seq"
    (a list of nodes), "scalar" (a string; ``style`` "plain", "single" or "double") or "null"
    (an empty value)."""

    __slots__ = ("kind", "value", "line", "col", "style")

    def __init__(self, kind: str, value: Any, line: int, col: int, style: str = "plain") -> None:
        self.kind, self.value, self.line, self.col, self.style = kind, value, line, col, style

    def get(self, key: str) -> Optional["Node"]:
        for k, _, v in self.value:
            if k == key:
                return v
        return None

    def keys(self) -> List[str]:
        return [k for k, _, _ in self.value]


class _Line:
    __slots__ = ("no", "indent", "text")

    def __init__(self, no: int, indent: int, text: str) -> None:
        self.no, self.indent, self.text = no, indent, text


_OPENERS = set(" \t[{,:-?")


def _strip_comment(s: str) -> str:
    quote = None
    prev = " "
    i = 0
    while i < len(s):
        c = s[i]
        if quote == "'":
            if c == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    i += 1
                else:
                    quote = None
        elif quote == '"':
            if c == "\\":
                i += 1
            elif c == '"':
                quote = None
        elif c in "'\"" and (prev in _OPENERS or not s[:i].strip()):
            quote = c
        elif c == "#" and prev in " \t":
            return s[:i]
        prev = c
        i += 1
    return s


def _lines(text: str) -> List[_Line]:
    out: List[_Line] = []
    started = False
    for no, raw in enumerate(text.split("\n"), 1):
        s = raw[:-1] if raw.endswith("\r") else raw
        if not started and not out and s.startswith("%"):
            continue  # (a directive)
        if s == "---" or s.startswith("--- ") or s.startswith("---\t"):
            if started or out:
                break  # (only the first document is read)
            started = True
            rest = _strip_comment(s[3:]).strip()
            if rest:
                out.append(_Line(no, 0, rest))  # (the root node, on the marker's line)
            continue
        if s == "..." or s.startswith("... "):
            break
        content = _strip_comment(s).rstrip()
        body = content.lstrip(" ")
        if not body:
            continue
        indent = len(content) - len(body)
        if body[0] == "\t":
            raise YamlError("tabs are not allowed for indentation", no, indent + 1)
        out.append(_Line(no, indent, body))
    return out


def _is_seq_entry(text: str) -> bool:
    return text == "-" or text.startswith("- ") or text.startswith("-\t")


def _quoted_end(s: str, i: int) -> int:
    """The index just past the quoted scalar starting at ``s[i]``, or -1 if it does not close."""
    q = s[i]
    j = i + 1
    while j < len(s):
        if q == "'" and s[j] == "'":
            if j + 1 < len(s) and s[j + 1] == "'":
                j += 2
                continue
            return j + 1
        if q == '"':
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == '"':
                return j + 1
        j += 1
    return -1


_ESCAPES = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n", "v": "\v", "f": "\f", "r": "\r", "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": " ", "P": " "}


def _unquote(raw: str, line: int, col: int) -> str:
    body = raw[1:-1]
    if raw[0] == "'":
        return body.replace("''", "'")
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        e = body[i + 1] if i + 1 < len(body) else ""
        width = {"x": 2, "u": 4, "U": 8}.get(e)
        if width:
            digits = body[i + 2 : i + 2 + width]
            if len(digits) != width or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise YamlError("invalid escape sequence", line, col + i + 1)
            out.append(chr(int(digits, 16)))
            i += 2 + width
        elif e in _ESCAPES:
            out.append(_ESCAPES[e])
            i += 2
        else:
            raise YamlError("unknown escape sequence", line, col + i + 1)
    return "".join(out)


class _Parser:
    def __init__(self, lines: List[_Line]) -> None:
        self.lines, self.i = lines, 0

    def peek(self) -> Optional[_Line]:
        return self.lines[self.i] if self.i < len(self.lines) else None

    def key(self, ln: _Line) -> Optional[Tuple[str, int, str, int]]:
        """(key, key column, the rest after ':', its column) if the line starts a mapping entry."""
        t = ln.text
        if t[0] in "'\"":
            end = _quoted_end(t, 0)
            if end < 0:
                return None
            j = end
            while j < len(t) and t[j] == " ":
                j += 1
            if j >= len(t) or t[j] != ":" or (j + 1 < len(t) and t[j + 1] not in " \t"):
                return None
            key = _unquote(t[:end], ln.no, ln.indent + 1)
        else:
            if t[0] in "[{#&*!|>%@`" or _is_seq_entry(t) or t[0] == "?":
                return None
            j = 0
            while True:
                j = t.find(":", j)
                if j < 0:
                    return None
                if j + 1 == len(t) or t[j + 1] in " \t":
                    break
                j += 1
            key = t[:j].rstrip()
            if not key:
                return None
        rest = t[j + 1 :]
        stripped = rest.lstrip(" \t")
        return key, ln.indent + 1, stripped, ln.indent + 1 + (j + 1) + (len(rest) - len(stripped))

    def block(self, min_indent: int) -> Node:
        ln = self.peek()
        if ln is None or ln.indent < min_indent:
            prev = self.lines[self.i - 1] if self.i else None
            return Node("null", None, prev.no if prev else 1, prev.indent + len(prev.text) + 1 if prev else 1)
        if _is_seq_entry(ln.text):
            return self.seq(ln.indent)
        if self.key(ln) is not None:
            return self.mapping(ln.indent)
        self.i += 1
        return self.inline(ln, ln.text, ln.indent + 1, min_indent - 1)

    def seq(self, indent: int) -> Node:
        first = self.peek()
        items: List[Node] = []
        while True:
            ln = self.peek()
            if ln is None or ln.indent != indent or not _is_seq_entry(ln.text):
                break
            rest = ln.text[1:]
            stripped = rest.lstrip(" \t")
            if not stripped:
                self.i += 1
                items.append(self.block(indent + 1))
                continue
            col = indent + 1 + (len(rest) - len(stripped))
            virtual = _Line(ln.no, col, stripped)
            self.lines[self.i] = virtual
            if _is_seq_entry(stripped):
                items.append(self.seq(col))
            elif self.key(virtual) is not None:
                items.append(self.mapping(col))
            else:
                self.i += 1
                items.append(self.inline(virtual, stripped, col + 1, indent))
        self._check_dedent(indent)
        return Node("seq", items, first.no, first.indent + 1)

    def mapping(self, indent: int) -> Node:
        first = self.peek()
        entries: List[Tuple[str, Node, Node]] = []
        while True:
            ln = self.peek()
            if ln is None or ln.indent != indent:
                break
            kv = self.key(ln)
            if kv is None:
                if _is_seq_entry(ln.text):
                    raise YamlError("sequence entries are not allowed here", ln.no, ln.indent + 1)
                raise YamlError("expected a mapping key", ln.no, ln.indent + 1)
            key, key_col, rest, rest_col = kv
            key_node = Node("scalar", key, ln.no, key_col)
            if any(k == key for k, _, _ in entries):
                raise YamlError(f"duplicated mapping key '{key}'", ln.no, key_col)
            self.i += 1
            if rest:
                value = self.inline(ln, rest, rest_col, indent)
            else:
                nxt = self.peek()
                if nxt is not None and nxt.indent > indent:
                    value = self.block(indent + 1)
                elif nxt is not None and nxt.indent == indent and _is_seq_entry(nxt.text):
                    value = self.seq(indent)
                else:
                    value = Node("null", None, ln.no, rest_col)
            entries.append((key, key_node, value))
        self._check_dedent(indent)
        return Node("map", entries, first.no, first.indent + 1)

    def _check_dedent(self, indent: int) -> None:
        ln = self.peek()
        if ln is not None and ln.indent > indent:
            raise YamlError("bad indentation", ln.no, ln.indent + 1)

    def inline(self, ln: _Line, text: str, col: int, parent_indent: int) -> Node:
        """The value ``text`` (from column ``col`` of line ``ln``), with any flow collection's or
        plain scalar's continuation lines (those indented past ``parent_indent``)."""
        c = text[0]
        if c in "[{":
            s, starts = text, [(0, ln.no, col)]
            while not _balanced(s):
                nxt = self.peek()
                if nxt is None:
                    raise YamlError(f"expected '{']' if c == '[' else '}'}' to close the flow collection", ln.no, col)
                self.i += 1
                starts.append((len(s) + 1, nxt.no, nxt.indent + 1))
                s += "\n" + nxt.text
            node, end = _Flow(s, starts).node(0, "")
            if s[end:].strip():
                raise YamlError("unexpected content after the flow collection", *_Flow(s, starts).pos(end))
            return node
        if c in "'\"":
            end = _quoted_end(text, 0)
            if end < 0:
                raise YamlError("unterminated quoted scalar", ln.no, col)
            if text[end:].strip():
                raise YamlError("unexpected content after the quoted scalar", ln.no, col + end)
            return Node("scalar", _unquote(text[:end], ln.no, col), ln.no, col, "single" if c == "'" else "double")
        if c in "|>":
            raise YamlError("block scalars are not supported", ln.no, col)
        if c in "&*!":
            raise YamlError("anchors, aliases and tags are not supported", ln.no, col)
        value = text.strip()
        while True:
            nxt = self.peek()
            if nxt is None or nxt.indent <= parent_indent or _is_seq_entry(nxt.text) or self.key(nxt) is not None:
                break
            value += " " + nxt.text.strip()
            self.i += 1
        if ": " in value or value.endswith(":") or ":\t" in value:
            raise YamlError("mapping values are not allowed in this context", ln.no, col)
        return Node("scalar", value, ln.no, col)


def _balanced(s: str) -> bool:
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in "'\"" and (i == 0 or s[i - 1] in " \t\n[{,:"):
            end = _quoted_end(s, i)
            if end < 0:
                return False
            i = end
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return True
        i += 1
    return depth <= 0


class _Flow:
    """A flow collection (``[ ... ]`` or ``{ ... }``), possibly spanning lines."""

    def __init__(self, s: str, starts: List[Tuple[int, int, int]]) -> None:
        self.s, self.starts = s, starts

    def pos(self, i: int) -> Tuple[int, int]:
        off, line, col = self.starts[0]
        for o, li, co in self.starts:
            if o <= i:
                off, line, col = o, li, co
        return line, col + (i - off)

    def err(self, msg: str, i: int) -> YamlError:
        return YamlError(msg, *self.pos(i))

    def ws(self, i: int) -> int:
        while i < len(self.s) and self.s[i] in " \t\n":
            i += 1
        return i

    def node(self, i: int, stops: str) -> Tuple[Node, int]:
        s = self.s
        i = self.ws(i)
        if i >= len(s):
            raise self.err("unexpected end of the flow collection", i)
        line, col = self.pos(i)
        c = s[i]
        if c == "[":
            items: List[Node] = []
            i = self.ws(i + 1)
            while True:
                if i >= len(s):
                    raise self.err("expected ']'", i)
                if s[i] == "]":
                    return Node("seq", items, line, col), i + 1
                item, i = self.node(i, ",]")
                items.append(item)
                i = self.ws(i)
                if i < len(s) and s[i] == ",":
                    i = self.ws(i + 1)
                elif i >= len(s) or s[i] != "]":
                    raise self.err("expected ',' or ']'", i)
        if c == "{":
            entries: List[Tuple[str, Node, Node]] = []
            i = self.ws(i + 1)
            while True:
                if i >= len(s):
                    raise self.err("expected '}'", i)
                if s[i] == "}":
                    return Node("map", entries, line, col), i + 1
                key_node, i = self.node(i, ",:}")
                if key_node.kind != "scalar":
                    raise self.err("flow mapping keys must be scalars", i)
                i = self.ws(i)
                if i < len(s) and s[i] == ":":
                    j = self.ws(i + 1)
                    if j < len(s) and s[j] not in ",}":
                        value, i = self.node(j, ",}")
                    else:
                        value, i = Node("null", None, *self.pos(i)), j
                else:
                    value = Node("null", None, *self.pos(i))
                if any(k == key_node.value for k, _, _ in entries):
                    raise YamlError(f"duplicated mapping key '{key_node.value}'", key_node.line, key_node.col)
                entries.append((key_node.value, key_node, value))
                i = self.ws(i)
                if i < len(s) and s[i] == ",":
                    i = self.ws(i + 1)
                elif i >= len(s) or s[i] != "}":
                    raise self.err("expected ',' or '}'", i)
        if c in "'\"":
            end = _quoted_end(s, i)
            if end < 0:
                raise self.err("unterminated quoted scalar", i)
            return Node("scalar", _unquote(s[i:end], line, col), line, col, "single" if c == "'" else "double"), end
        if c in "&*!|>":
            raise self.err("anchors, aliases, tags and block scalars are not supported", i)
        # A plain scalar: up to a flow indicator (or, for a key, a ':' ending it).
        j = i
        while j < len(s) and s[j] not in ",[]{}":
            if s[j] == ":" and ":" in stops and (j + 1 == len(s) or s[j + 1] in " \t\n,]}"):
                break
            j += 1
        text = " ".join(s[i:j].split())
        if not text:
            return Node("null", None, line, col), j
        return Node("scalar", text, line, col), j


def parse(text: str) -> Optional[Node]:
    """The first document's root node (None for an empty document)."""
    lines = _lines(text)
    if not lines:
        return None
    p = _Parser(lines)
    root = p.block(0)
    ln = p.peek()
    if ln is not None:
        raise YamlError("unexpected content", ln.no, ln.indent + 1)
    return None if root.kind == "null" else root


# -- Writing --------------------------------------------------------------------------------------

# Emit values: ("scalar", text), ("flow", [texts]), ("seq", [values]), ("map", [(key, value)]).


def _is_numeric(s: str) -> bool:
    # LLVM's yaml::isNumeric (YAML 1.2 core schema numbers).
    if not s or s in ("+", "-"):
        return False
    if s in (".nan", ".NaN", ".NAN"):
        return True
    tail = s[1:] if s[0] in "+-" else s
    if tail in (".inf", ".Inf", ".INF"):
        return True
    if s.startswith("0o"):
        return len(s) > 2 and all(c in "01234567" for c in s[2:])
    if s.startswith("0x"):
        return len(s) > 2 and all(c in "0123456789abcdefABCDEF" for c in s[2:])
    s = tail
    if s.startswith(".") and (s == "." or (len(s) > 1 and s[1] not in "0123456789")):
        return False
    if s[:1] in ("e", "E"):
        return False

    def digits(t: str) -> str:
        return t.lstrip("0123456789")

    s = digits(s)
    if not s:
        return True
    if s[0] == ".":
        s = digits(s[1:])
        if not s:
            return True
        if s[0] not in "eE":
            return False
    elif s[0] not in "eE":
        return False
    s = s[1:]
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
        if not s:
            return False
    return not digits(s)


def needs_quotes(s: str) -> str:
    """LLVM's yaml::needsQuotes: "none", "single" or "double"."""
    if not s:
        return "single"
    need = "none"
    if s[0] in " \t\n\r\v\f" or s[-1] in " \t\n\r\v\f":
        need = "single"
    if s in ("null", "Null", "NULL", "~") or s in ("true", "True", "TRUE", "false", "False", "FALSE") or _is_numeric(s):
        need = "single"
    if s[0] in "-?:\\,[]{}#&*!|>'\"%@`":
        need = "single"
    for ch in s.encode("utf-8"):
        if (48 <= ch <= 57) or (65 <= ch <= 90) or (97 <= ch <= 122):
            continue
        if ch in b"_-^., \t":
            continue
        if ch in (0x0A, 0x0D):
            need = "single"
            continue
        if ch == 0x7F or ch <= 0x1F or ch & 0x80:
            return "double"
        need = "single"
    return need


def _escape_double(s: str) -> str:
    # llvm::yaml::escape, EscapePrintable false.
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\0":
            out.append("\\0")
        elif ch == "\a":
            out.append("\\a")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\v":
            out.append("\\v")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\x1b":
            out.append("\\e")
        elif o < 0x20:
            out.append("\\x" + format(o, "02X"))
        elif o == 0x85:
            out.append("\\N")
        elif o == 0xA0:
            out.append("\\_")
        elif o == 0x2028:
            out.append("\\L")
        elif o == 0x2029:
            out.append("\\P")
        else:
            out.append(ch)
    return "".join(out)


def quote(s: str) -> str:
    """A string scalar as LLVM writes it."""
    if not s:
        return "''"
    need = needs_quotes(s)
    if need == "none":
        return s
    if need == "single":
        return "'" + s.replace("'", "''") + "'"
    return '"' + _escape_double(s) + '"'


def fmt_double(v: float) -> str:
    """A double as LLVM writes it (printf ``%g``)."""
    return "%g" % v


def _pad(key: str) -> str:
    return " " * (16 - len(key)) if len(key) < 16 else " "


def _flow(head: str, elems: List[str], wrap: int) -> List[str]:
    start = len(head)  # (the column of '[')
    s, col, lines = head + "[ ", start + 2, []
    for k, e in enumerate(elems):
        if k:
            s += ", "
            col += 2
        if wrap and col > wrap:
            lines.append(s)
            s, col = " " * start + "  ", start + 2
        s += e
        col += len(e)
    lines.append(s + " ]")
    return lines


def _map(items: List[Tuple[str, Any]], indent: int, wrap: int, first: Optional[str] = None) -> List[str]:
    lines: List[str] = []
    for k, (key, (kind, val)) in enumerate(items):
        head = (first if k == 0 and first is not None else " " * indent) + key + ":"
        if kind == "scalar":
            lines.append(head + _pad(key) + val)
        elif kind == "flow":
            lines.extend(_flow(head + _pad(key), val, wrap))
        elif not val:
            lines.append(head + _pad(key) + ("[]" if kind == "seq" else "{}"))
        else:
            lines.append(head)
            lines.extend(_seq(val, indent + 2, wrap) if kind == "seq" else _map(val, indent + 2, wrap))
    return lines


def _seq(items: List[Any], indent: int, wrap: int) -> List[str]:
    lines: List[str] = []
    for kind, val in items:
        dash = " " * indent + "- "
        if kind == "scalar":
            lines.append(dash + val)
        elif kind == "flow":
            lines.extend(_flow(dash, val, wrap))
        elif not val:
            lines.append(dash + ("[]" if kind == "seq" else "{}"))
        elif kind == "map":
            lines.extend(_map(val, indent + 2, wrap, first=dash))
        else:
            inner = _seq(val, indent + 2, wrap)
            lines.extend([dash + inner[0][indent + 2 :]] + inner[1:])
    return lines


def document(items: List[Tuple[str, Any]], wrap: int) -> str:
    """A document whose root is the mapping ``items``, as llvm::yaml::Output writes it."""
    return "---\n" + "\n".join(_map(items, 0, int(wrap))) + "\n...\n"
