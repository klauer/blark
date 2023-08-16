from typing import Optional

import pytest

from ..dependency_store import DependencyStore
from . import conftest

if not conftest.DS_CONFIG.exists():
    pytest.skip(
        "twincat_root directory not found! Did you recursively clone the "
        "repository? (git clone --recursive ...)",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "project_name, version",
    [
        ("LCLS_General", "v2.8.1"),
        ("project_a", "*"),
        ("project_b", "*"),
    ]
)
def test_load_project(
    store: DependencyStore,
    project_name: str,
    version: Optional[str],
):
    assert project_name in store.config.libraries
    matches = store.get_dependency(project_name, version)
    try:
        match, = matches
    except ValueError:
        raise RuntimeError(f"No match for project {project_name} version={version}")
