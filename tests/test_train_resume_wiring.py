import ast
from pathlib import Path


def test_main_passes_resume_payload_into_train():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    train_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train"
    ]
    assert len(train_calls) == 1
    resume_keywords = [
        keyword
        for keyword in train_calls[0].keywords
        if keyword.arg == "resume_payload"
    ]
    assert len(resume_keywords) == 1
    assert isinstance(resume_keywords[0].value, ast.Name)
    assert resume_keywords[0].value.id == "resume_payload"
