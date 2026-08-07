#!/usr/bin/env python3
"""Strict AST-based placeholder test scanner.

Reports test functions that:
  - contain only `pass`
  - contain only a docstring
  - contain only `...`
  - contain unconditional pytest.skip
  - contain unconditional pytest.xfail
  - contain no assertion and do not call a helper that asserts

Exit code: 0 if zero placeholders, 1 otherwise.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _is_bare_literal(node: ast.stmt) -> bool:
    """True if node is `pass`, `...`, or a bare string literal (docstring)."""
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr):
        val = node.value
        # `...` is Constant(value=Ellipsis)
        if isinstance(val, ast.Constant) and val.value is ...:
            return True
        # bare string literal = docstring
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return True
    return False


def _is_unconditional_pytest_call(node: ast.stmt, names: set[str]) -> bool:
    """True if node is a bare call like pytest.skip(...) or pytest.xfail(...)."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Attribute):
        return (
            isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
            and func.attr in names
        )
    if isinstance(func, ast.Name):
        return func.id in {f"pytest.{n}" for n in names}
    return False


def _has_assertion(body: list[ast.stmt]) -> bool:
    """Recursively search for any assert statement, or a call whose name
    contains 'assert' (e.g. assertTrue, assertEqual, assertClose)."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if "assert" in name.lower() or "check" in name.lower():
                return True
    return False


def _classify_function(func: ast.FunctionDef) -> str | None:
    """Return a string describing why this is a placeholder, or None if OK."""
    if not func.name.startswith("test_"):
        return None

    body = func.body

    # Strip leading docstring
    effective_body = body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        effective_body = body[1:]

    # Completely empty after docstring
    if not effective_body:
        return "empty-after-docstring"

    # Only pass / ... / docstring
    if all(_is_bare_literal(n) for n in effective_body):
        return "body-is-only-pass-or-ellipsis-or-docstring"

    # Unconditional skip / xfail
    for node in effective_body:
        if _is_unconditional_pytest_call(node, {"skip", "xfail"}):
            return "unconditional-pytest-skip-or-xfail"

    # No assertion at all
    if not _has_assertion(effective_body):
        return "no-assertion-found"

    return None


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (func_name, line_number, reason) placeholders."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            reason = _classify_function(node)
            if reason is not None:
                issues.append((node.name, node.lineno, reason))
    return issues


def main() -> int:
    files = sys.argv[1:]
    if not files:
        print("Usage: check_test_placeholders.py <file1.py> [file2.py ...]")
        return 1

    total_issues = 0
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            return 1
        issues = scan_file(path)
        if issues:
            print(f"\n=== {path} ===")
            for func_name, lineno, reason in issues:
                print(f"  PLACEHOLDER  {func_name}  (line {lineno})  [{reason}]")
            total_issues += len(issues)
        else:
            print(f"OK: {path} — 0 placeholder tests")

    if total_issues == 0:
        print("\nRESULT: 0 placeholder tests, 0 unconditional skips, 0 unconditional xfails")
        return 0
    else:
        print(f"\nRESULT: {total_issues} placeholder test(s) found — FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
