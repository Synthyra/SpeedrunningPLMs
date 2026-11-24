"""The documents against the code: program.md's hook tables name what strategy.py defines, its proposers are
search.py's, and the public README's record table agrees with results.md. Read with `ast`, so no model is built."""

from __future__ import annotations

import ast
import re

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
STRATEGY = PROJECT / "src" / "speedrunning_plms" / "strategies" / "strategy.py"
RECORD_ROWS = {"10 minutes": "record, 10 min", "1 hour": "record, 1 hour"}  # PUBLIC_README's track -> results.md's model


def read(name: str) -> str:
    return (PROJECT / name).read_text(encoding="utf-8")


def module_constant(path: Path, name: str) -> tuple[str, ...]:
    """A module-level tuple of string literals, read from the source without importing it."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{path.name} defines no {name}")


def class_methods(path: Path, name: str) -> set[str]:
    """The functions and properties a class defines in its body."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
    raise AssertionError(f"{path.name} defines no class {name}")


def table_rows(text: str, header: str) -> list[list[str]]:
    """The cells of each row of the Markdown table whose header line is exactly `header`."""
    lines = text.splitlines()
    start = lines.index(header)
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def code_name(cell: str) -> str:
    match = re.fullmatch(r"`(\w+)`", cell)
    assert match is not None, f"{cell!r} is not one code span"
    return match.group(1)


def number(cell: str) -> float:
    return float(cell.strip("*"))


def test_the_hook_table_names_every_hook_in_order() -> None:
    rows = table_rows(read("program.md"), "| Hook | When it runs | In the compiled model |")
    assert tuple(code_name(row[0]) for row in rows) == module_constant(STRATEGY, "HOOKS")
    assert all(row[2] in ("yes", "no") for row in rows)


def test_the_method_table_names_strategy_methods_that_are_not_hooks() -> None:
    rows = table_rows(read("program.md"), "| Method | When it runs |")
    names = {code_name(row[0]) for row in rows}
    assert names <= class_methods(STRATEGY, "Strategy")
    assert not names & set(module_constant(STRATEGY, "HOOKS"))


def test_program_names_every_proposer_and_no_other() -> None:
    named = set(re.findall(r"--proposer (\w+)", read("program.md")))
    assert named == set(module_constant(PROJECT / "search.py", "PROPOSER_NAMES"))


def test_the_public_record_table_agrees_with_results() -> None:
    results = read("results.md")
    references = {row[0].strip("*"): row for row in table_rows(
        results, "| Model | Parameters | `val_select` | Validation | Test loss | Bits | Accuracy | F1 macro | MCC | ROC AUC | PR AUC |")}
    records = table_rows(read("PUBLIC_README.md"), "| Track | Training clock | Test loss | `val_select` | Genome |")
    assert {row[0] for row in records} == set(RECORD_ROWS)
    for track, _, test_loss, val_select, genome in records:
        reference = references[RECORD_ROWS[track]]
        assert (number(test_loss), number(val_select)) == (number(reference[4]), number(reference[2])), track
        path = re.fullmatch(r"\[`(\w+)`\]\((genomes/\w+\.json)\)", genome)
        assert path is not None and (PROJECT / path.group(2)).is_file(), genome
        assert path.group(2) in results, f"results.md does not name {path.group(2)}"
