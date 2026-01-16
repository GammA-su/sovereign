from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..utils import stable_hash
from .dsl import Expr, Program

INTERPRETER_VERSION = "v1"
INTERPRETER_HASH = stable_hash({"version": INTERPRETER_VERSION, "dsl": "bvps"})


@dataclass
class TraceStep:
    step: int
    op: str
    state_hash: str


@dataclass
class EvalResult:
    value: Any
    trace: List[TraceStep]
    trace_hash: str
    steps: int


@dataclass
class Interpreter:
    step_limit: int = 500
    memory_limit: int = 1000
    _steps: int = 0
    trace: List[TraceStep] = field(default_factory=list)

    def _tick(self, op: str, env: Dict[str, Any]) -> None:
        self._steps += 1
        if self._steps > self.step_limit:
            raise RuntimeError("step limit exceeded")
        state_hash = stable_hash({"op": op, "env": env, "step": self._steps})
        self.trace.append(TraceStep(step=self._steps, op=op, state_hash=state_hash))

    def _eval_expr(self, expr: Expr, env: Dict[str, Any]) -> Any:
        kind = expr["kind"]
        self._tick(kind, env)
        if kind == "int":
            return int(expr["value"])
        if kind == "bool":
            return bool(expr["value"])
        if kind == "var":
            return env[expr["name"]]
        if kind == "binop":
            left = self._eval_expr(expr["left"], env)
            right = self._eval_expr(expr["right"], env)
            op = expr["op"]
            if op == "+":
                return int(left) + int(right)
            if op == "-":
                return int(left) - int(right)
            if op == "*":
                return int(left) * int(right)
            if op == "==":
                return left == right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "and":
                return bool(left) and bool(right)
            if op == "or":
                return bool(left) or bool(right)
            raise ValueError(f"unknown op {op}")
        if kind == "if":
            cond = self._eval_expr(expr["cond"], env)
            branch = expr["then"] if cond else expr["else"]
            return self._eval_expr(branch, env)
        if kind == "let":
            value = self._eval_expr(expr["value"], env)
            new_env = dict(env)
            new_env[expr["name"]] = value
            return self._eval_expr(expr["body"], new_env)
        if kind == "list":
            values = [self._eval_expr(item, env) for item in expr["elements"]]
            if len(values) > self.memory_limit:
                raise RuntimeError("memory limit exceeded")
            return values
        if kind == "tuple2":
            return (
                self._eval_expr(expr["left"], env),
                self._eval_expr(expr["right"], env),
            )
        if kind == "map":
            items = self._eval_expr(expr["list"], env)
            if not isinstance(items, list):
                raise ValueError("map expects list")
            bound = int(expr["bound"])
            if len(items) > bound:
                raise RuntimeError("map bound exceeded")
            result: list[Any] = []
            for item in items:
                if len(result) > self.memory_limit:
                    raise RuntimeError("memory limit exceeded")
                new_env = dict(env)
                new_env[expr["var"]] = item
                result.append(self._eval_expr(expr["body"], new_env))
            return result
        if kind == "fold":
            items = self._eval_expr(expr["list"], env)
            if not isinstance(items, list):
                raise ValueError("fold expects list")
            bound = int(expr["bound"])
            if len(items) > bound:
                raise RuntimeError("fold bound exceeded")
            acc = self._eval_expr(expr["init"], env)
            for item in items:
                new_env = dict(env)
                new_env[expr["var"]] = item
                new_env[expr["acc"]] = acc
                acc = self._eval_expr(expr["body"], new_env)
            return acc
        raise ValueError(f"unknown expr kind {kind}")

    def evaluate(self, program: Program, inputs: Dict[str, Any]) -> EvalResult:
        self._steps = 0
        self.trace = []
        for cond in program.pre:
            if not self._eval_expr(cond, inputs):
                raise ValueError("precondition failed")
        value = self._eval_expr(program.body, inputs)
        for cond in program.post:
            if not self._eval_expr(cond, {**inputs, "result": value}):
                raise ValueError("postcondition failed")
        trace_hash = stable_hash([step.__dict__ for step in self.trace])
        return EvalResult(value=value, trace=self.trace, trace_hash=trace_hash, steps=self._steps)


def eval_program(program: Program, inputs: Dict[str, Any], step_limit: int) -> Tuple[Any, str]:
    interpreter = Interpreter(step_limit=step_limit)
    result = interpreter.evaluate(program, inputs)
    return result.value, result.trace_hash
