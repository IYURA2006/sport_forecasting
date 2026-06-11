"""Scaffold smoke test: the package and every planned subpackage import cleanly."""

import importlib

import pytest

SUBPACKAGES = ["data", "models", "inference", "markets", "eval", "sim", "report"]


def test_package_imports() -> None:
    import wc26

    assert wc26.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"wc26.{name}")
