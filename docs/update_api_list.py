"""
Tool that generates the source for ``source/api.rst``.
"""
from __future__ import annotations

import inspect
import pathlib
import sys
from types import ModuleType
from typing import Callable, Optional

docs_path = pathlib.Path(__file__).parent.resolve()
module_path = docs_path.parent
sys.path.insert(0, str(module_path))

import blark  # noqa: E402
import blark.apischema_compat  # noqa: E402
import blark.config  # noqa: E402
import blark.dependency_store  # noqa: E402
import blark.format  # noqa: E402
import blark.html  # noqa: E402
import blark.input  # noqa: E402
import blark.main  # noqa: E402
import blark.output  # noqa: E402
import blark.parse  # noqa: E402
import blark.plain  # noqa: E402
import blark.solution  # noqa: E402
import blark.sphinxdomain  # noqa: E402
import blark.summary  # noqa: E402
import blark.transform  # noqa: E402
import blark.typing  # noqa: E402
import blark.util  # noqa: E402


def find_all_classes(
    modules,
    base_classes: tuple[type, ...],
    skip: Optional[list[str]] = None,
) -> list[type]:
    """Find all classes in the module and return them as a list."""
    skip = skip or []

    def should_include(obj):
        return (
            inspect.isclass(obj) and
            (not base_classes or issubclass(obj, base_classes)) and
            obj.__name__ not in skip
        )

    def sort_key(cls):
        return (cls.__module__, cls.__name__)

    classes = [
        obj
        for module in modules
        for _, obj in inspect.getmembers(module, predicate=should_include)
    ]

    return list(sorted(set(classes), key=sort_key))


def find_callables(modules: list[ModuleType]) -> list[Callable]:
    """Find all callables in the module and return them as a list."""
    def should_include(obj):
        try:
            name = obj.__name__
            module = obj.__module__
        except AttributeError:
            return False

        if not any(module.startswith(mod.__name__) for mod in modules):
            return False

        return (
            callable(obj) and
            not inspect.isclass(obj) and
            not name.startswith("_")
        )

    def sort_key(obj):
        return (obj.__module__, obj.__name__)

    callables = [
        obj
        for module in modules
        for _, obj in inspect.getmembers(module, predicate=should_include)
    ]

    return list(sorted(set(callables), key=sort_key))


def create_api_list(modules: list[ModuleType]) -> list[str]:
    """Create the API list with all classes and functions."""
    output = [
        "API",
        "###",
        "",
    ]

    for module in modules:
        classes = find_all_classes([module], base_classes=())
        callables = find_callables([module])
        module_name = module.__name__
        underline = "-" * len(module_name)
        output.append(module_name)
        output.append(underline)
        output.append("")
        objects = [
            obj
            for obj in list(classes) + list(callables)
            if obj.__module__ == module_name and hasattr(module, obj.__name__)
        ]

        if objects:
            output.append(".. autosummary::")
            output.append("    :toctree: api")
            output.append("")

            for obj in sorted(objects, key=lambda obj: obj.__name__):
                output.append(f"    {obj.__module__}.{obj.__name__}")

            output.append("")

    while output[-1] == "":
        output.pop(-1)
    return output


if __name__ == "__main__":
    output = create_api_list(
        [
            blark.apischema_compat,
            blark.config,
            blark.dependency_store,
            blark.format,
            blark.html,
            blark.input,
            blark.main,
            blark.output,
            blark.parse,
            blark.plain,
            blark.solution,
            blark.sphinxdomain,
            blark.summary,
            blark.transform,
            blark.typing,
            blark.util,
        ]
    )
    print("\n".join(output))
