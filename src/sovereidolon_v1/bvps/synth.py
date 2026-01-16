from __future__ import annotations

from typing import Iterable, List

from ..orchestrator.task import Task
from .dsl import Program, binop, fold_list, if_expr, lit_int, var


def _list_bound(task: Task, default: int = 10) -> int:
    bound = task.bounds.get("xs_len") or task.bounds.get("list_len")
    if isinstance(bound, list) and bound:
        return int(bound[-1])
    return default


def program_identity(task: Task) -> Program:
    arg = next(iter(task.inputs.keys()))
    return Program(
        name="identity",
        arg_types=task.inputs,
        return_type=task.output,
        body=var(arg),
    )


def program_add(task: Task) -> Program:
    arg_names = list(task.inputs.keys())
    return Program(
        name="add",
        arg_types=task.inputs,
        return_type=task.output,
        body=binop("+", var(arg_names[0]), var(arg_names[1])),
    )


def program_mul(task: Task) -> Program:
    arg_names = list(task.inputs.keys())
    return Program(
        name="mul",
        arg_types=task.inputs,
        return_type=task.output,
        body=binop("*", var(arg_names[0]), var(arg_names[1])),
    )


def program_list_sum(task: Task) -> Program:
    bound = _list_bound(task)
    return Program(
        name="sum",
        arg_types=task.inputs,
        return_type=task.output,
        body=fold_list(
            var_name="x",
            acc_name="acc",
            list_expr=var("xs"),
            init=lit_int(0),
            body=binop("+", var("acc"), var("x")),
            bound=bound,
        ),
    )


def program_list_max(task: Task) -> Program:
    bound = _list_bound(task)
    return Program(
        name="max",
        arg_types=task.inputs,
        return_type=task.output,
        body=fold_list(
            var_name="x",
            acc_name="acc",
            list_expr=var("xs"),
            init=lit_int(-10**6),
            body=if_expr(
                binop(">", var("x"), var("acc")),
                var("x"),
                var("acc"),
            ),
            bound=bound,
        ),
    )


def candidate_programs(task: Task) -> Iterable[Program]:
    if task.task_type == "arith":
        if task.goal == "add":
            return [program_identity(task), program_add(task), program_mul(task)]
        if task.goal == "mul":
            return [program_identity(task), program_mul(task)]
        return [program_identity(task)]
    if task.task_type == "list":
        if task.goal == "sum":
            return [program_identity(task), program_list_sum(task)]
        if task.goal == "max":
            return [program_identity(task), program_list_max(task)]
        return [program_identity(task)]
    return [program_identity(task)]


def list_candidate_names(task: Task) -> List[str]:
    return [prog.name for prog in candidate_programs(task)]
