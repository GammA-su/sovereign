from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

Expr = Dict[str, Any]


class Program(BaseModel):
    name: str
    arg_types: Dict[str, str]
    return_type: str
    body: Expr
    pre: List[Expr] = Field(default_factory=list)
    post: List[Expr] = Field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return self.model_dump()


def lit_int(value: int) -> Expr:
    return {"kind": "int", "value": value}


def lit_bool(value: bool) -> Expr:
    return {"kind": "bool", "value": value}


def var(name: str) -> Expr:
    return {"kind": "var", "name": name}


def binop(op: str, left: Expr, right: Expr) -> Expr:
    return {"kind": "binop", "op": op, "left": left, "right": right}


def if_expr(cond: Expr, then: Expr, otherwise: Expr) -> Expr:
    return {"kind": "if", "cond": cond, "then": then, "else": otherwise}


def let(name: str, value: Expr, body: Expr) -> Expr:
    return {"kind": "let", "name": name, "value": value, "body": body}


def list_lit(elements: List[Expr]) -> Expr:
    return {"kind": "list", "elements": elements}


def tuple2(left: Expr, right: Expr) -> Expr:
    return {"kind": "tuple2", "left": left, "right": right}


def map_list(var_name: str, list_expr: Expr, body: Expr, bound: int) -> Expr:
    return {"kind": "map", "var": var_name, "list": list_expr, "body": body, "bound": bound}


def fold_list(
    var_name: str, acc_name: str, list_expr: Expr, init: Expr, body: Expr, bound: int
) -> Expr:
    return {
        "kind": "fold",
        "var": var_name,
        "acc": acc_name,
        "list": list_expr,
        "init": init,
        "body": body,
        "bound": bound,
    }
