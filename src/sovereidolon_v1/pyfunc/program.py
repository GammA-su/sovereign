from __future__ import annotations

import ast
from dataclasses import dataclass

from ..utils import hash_bytes

ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "len",
    "max",
    "min",
    "range",
    "sum",
}
DENIED_CALLS = {"eval", "exec", "open", "__import__", "globals", "locals", "vars", "dir"}

ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Expr,
    ast.Assign,
    ast.If,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Call,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.keyword,
    ast.Pass,
    ast.IfExp,
    ast.Subscript,
    ast.Slice,
    ast.UnaryOp,
    ast.And,
    ast.Or,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.While,
)

PYEXEC_INTERPRETER_HASH = "pyexec-v1"


class PyFuncValidationError(Exception):
    def __init__(self, failure_atom: str) -> None:
        super().__init__(failure_atom)
        self.failure_atom = failure_atom


@dataclass
class PyFuncProgram:
    code: str

    def __post_init__(self) -> None:
        canonical = canonical_pyfunc_source(self.code)
        object.__setattr__(self, "code", canonical)

    def to_json(self) -> dict[str, str]:
        return {"code": self.code}

    def to_bytes(self) -> bytes:
        return canonical_pyfunc_bytes(self.code)


def canonical_pyfunc_source(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n")
    return normalized + "\n"


def canonical_pyfunc_bytes(code: str) -> bytes:
    return canonical_pyfunc_source(code).encode("utf-8")


def compute_pyfunc_hash(code: str) -> str:
    return hash_bytes(canonical_pyfunc_bytes(code))


class _SafePyFuncVisitor(ast.NodeVisitor):
    def __init__(self, entrypoint: str) -> None:
        self.entrypoint = entrypoint
        self.found_entry = False
        super().__init__()

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise PyFuncValidationError("AST_FORBIDDEN_NODE")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        raise PyFuncValidationError("AST_FORBIDDEN_NODE")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise PyFuncValidationError("AST_FORBIDDEN_NODE")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise PyFuncValidationError("AST_FORBIDDEN_NODE")

    def visit_Global(self, node: ast.Global) -> None:
        raise PyFuncValidationError("AST_FORBIDDEN_NODE")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise PyFuncValidationError("AST_FORBIDDEN_NODE")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in DENIED_CALLS:
                raise PyFuncValidationError("AST_FORBIDDEN_CALL")
        else:
            raise PyFuncValidationError("AST_FORBIDDEN_NODE")
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.entrypoint:
            self.found_entry = True
        super().generic_visit(node)


def validate_pyfunc_code(code: str, entrypoint: str) -> None:
    tree = ast.parse(code)
    visitor = _SafePyFuncVisitor(entrypoint)
    visitor.visit(tree)
    if not visitor.found_entry:
        raise PyFuncValidationError("EXCEPTION:ENTRYPOINT_MISSING")
