"""
TwinCAT project dependency handling.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import packaging.version

from . import parse, util
from .config import BLARK_TWINCAT_ROOT
from .solution import (DependencyVersion, TwincatPlcProject,
                       make_solution_from_files)
from .summary import CodeSummary

AnyPath = Union[str, pathlib.Path]

logger = logging.getLogger(__name__)
_dependency_store = None


@dataclass
class DependencyStoreConfig:
    """Dependency store configuration, from ``config.json``."""

    filename: Optional[pathlib.Path]
    libraries: Dict[str, DependencyStoreLibrary]

    @classmethod
    def from_dict(
        cls, config: dict, filename: Optional[pathlib.Path] = None
    ) -> DependencyStoreConfig:
        libraries = config.get("libraries", {})
        for library_name, library_info in libraries.items():
            libraries[library_name] = DependencyStoreLibrary(**library_info)
        return cls(filename=filename, libraries=libraries)

    def as_json(self) -> str:
        """Get the configuration as JSON."""
        config = dataclasses.asdict(self)
        config.pop("filename")
        return json.dumps(config, indent=4)

    def save(self, path: AnyPath) -> None:
        """Save the configuration as JSON to a file."""
        with open(path, "wt") as fp:
            print(self.as_json(), file=fp)


@dataclass
class DependencyStoreLibrary:
    name: str
    versioned: bool
    path: str
    project: str

    def get_latest_version_path(self, root: pathlib.Path) -> pathlib.Path:
        """
        Get the latest version project filename.

        Returns
        -------
        pathlib.Path
        """

        def get_version(path: pathlib.Path) -> Optional[packaging.version.Version]:
            try:
                version_string = path.name.lstrip("v").replace("-", ".")
                return packaging.version.parse(version_string)
            except Exception:
                return None

        project_root = root / self.path

        paths = {
            (get_version(path), path)
            for path in project_root.iterdir()
            if get_version(path) is not None
        }

        for version, path in reversed(sorted(paths)):
            project_fn = path / self.project
            if project_fn.exists():
                logger.debug("Found latest %s %s in %s", self.name, version, project_fn)
                return project_fn

        raise FileNotFoundError(
            f"No valid versions of {self.name} found in {project_root}"
        )

    def get_project_filename(
        self,
        root: pathlib.Path,
        version: Optional[str],
    ) -> pathlib.Path:
        """Get the full project filename, given the root path and version."""
        if not self.versioned:
            return root / self.path / self.project
        if version == "*" or version is None:
            return self.get_latest_version_path(root)

        return root / self.path / version / self.project


class DependencyStore:
    """
    A storage container for dependency configuration and loading.

    Environment variable: ``BLARK_TWINCAT_ROOT`` is required to be set for this
    to be functional, along with a "config.json" in that directory.  This
    should contain information as to the supported library dependencies and
    where to find them.

    .. code::

        {
            "libraries": {
                "LCLS General": {
                    "name": "LCLS General",
                    "versioned": false,
                    "path": "lcls-twincat-general",
                    "project": "LCLSGeneral.sln"
                },
                "lcls-twincat-motion": {
                    "name": "lcls-twincat-motion",
                    "versioned": true,
                    "path": "lcls-twincat-motion",
                    "project": "lcls-twincat-motion.sln"
                }
            }
        }

    The above would indicate that the "LCLS General" library
    (as named in TwinCAT) is available relative to the root directory in
    ``lcls-twincat-general/LCLSGeneral.sln``.
    It would also indicate that the "lcls-twincat-motion" library could
    be found in
    ``lcls-twincat-motion/VERSION/lcls-twincat-motion.sln``
    where VERSION is the project-defined version.
    """

    root: pathlib.Path
    config: DependencyStoreConfig

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.load_config()

    @property
    def config_filename(self):
        """The configuration filename."""
        return (self.root / "config.json").expanduser().resolve()

    def _read_config(self) -> Any:
        with open(self.config_filename) as fp:
            return json.load(fp)

    def load_config(self):
        """Load the dependency store configuration file."""
        logger.debug("Loading dependency store config from %s", self.config_filename)
        try:
            config = self._read_config()
        except FileNotFoundError:
            logger.warning(
                "Project dependencies will not be loaded as either "
                "BLARK_TWINCAT_ROOT is unset or invalid.  Expected "
                "file %s to exist",
                self.root / "config.json",
            )
            self.config = DependencyStoreConfig(filename=None, libraries={})
            return

        self.config = DependencyStoreConfig.from_dict(
            config, filename=self.config_filename
        )

    @functools.lru_cache(maxsize=50)
    def get_dependency(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> List[PlcProjectMetadata]:
        """Get a dependency by name and version number."""
        try:
            info: DependencyStoreLibrary = self.config.libraries[name]
        except KeyError:
            logger.warning(
                "Unable to find library %s in dependency store. Known libraries: %s",
                name,
                ", ".join(self.config.libraries),
            )
            return []

        try:
            filename = info.get_project_filename(self.root, version=version)
        except FileNotFoundError:
            logger.warning(
                "Unable to find library project %s version %s", name, version
            )
            return []

        if not filename.exists():
            logger.warning(
                "Library project %s version %s file %s does not exist",
                name,
                version,
                filename,
            )
            return []

        return list(
            PlcProjectMetadata.from_project_filename(
                str(filename.resolve()),
                # TODO: one level only for now to avoid circular deps
                include_dependencies=False,
            )
        )

    def get_dependencies(
        self,
        plc: TwincatPlcProject,
    ) -> Generator[Tuple[DependencyVersion, PlcProjectMetadata], None, None]:
        """Get dependency projects from a PLC."""
        for _, info in plc.dependencies.items():
            if info.resolution is None:
                continue

            for proj in self.get_dependency(info.name, info.resolution.version):
                yield info.resolution, proj

    @staticmethod
    def get_instance() -> DependencyStore:
        """Get the global DependencyStore instance."""
        return get_dependency_store()


def get_dependency_store() -> DependencyStore:
    """Get the global DependencyStore instance."""
    global _dependency_store

    if _dependency_store is None:
        _dependency_store = DependencyStore(root=pathlib.Path(BLARK_TWINCAT_ROOT))
    return _dependency_store


@dataclass
class PlcProjectMetadata:
    """This is a per-PLC project metadata container."""

    name: str
    filename: pathlib.Path
    include_dependencies: bool
    code: List[parse.ParseResult]
    summary: CodeSummary
    loaded_files: Dict[pathlib.Path, str]
    dependencies: Dict[str, DependencyVersion]
    plc: Optional[TwincatPlcProject]

    @classmethod
    def from_plcproject(
        cls,
        plc: TwincatPlcProject,
        include_dependencies: bool = True,
    ) -> Optional[PlcProjectMetadata]:
        """Create a PlcProjectMetadata instance from a ``TwincatPlcProject``."""
        if plc.plcproj_path is None:
            raise ValueError(f"The PLC {plc.name!r} must have a location on disk.")

        filename = plc.plcproj_path.resolve()
        loaded_files = {}
        deps = {}
        code = []
        combined_summary = CodeSummary()

        loaded_files[filename] = util.get_file_sha256(filename)

        if include_dependencies:
            store = get_dependency_store()
            for resolution, proj in store.get_dependencies(plc):
                code.extend(proj.code)
                deps.update(proj.dependencies)
                loaded_files.update(proj.loaded_files)
                deps[resolution.name] = resolution
                plc_name = proj.plc.name if proj.plc is not None else "unknown"
                combined_summary.append(proj.summary, namespace=plc_name)

        for source in plc.sources:
            if not source.contents:
                continue

            for item in source.contents.to_blark():
                results = []
                for result in parse.parse_item(item):
                    if result.exception is not None:
                        logger.debug(
                            "Failed to load: %s %s", result.filename, result
                        )
                        continue
                    assert result.filename is not None
                    results.append(result)
                    loaded_files[result.filename] = util.get_file_sha256(
                        result.filename
                    )

                if results:
                    summary = CodeSummary.from_parse_results(results)
                    combined_summary.append(summary)

        # tmc = plc.tmc
        return cls(
            name=plc.name or "unknown",
            filename=filename,
            include_dependencies=include_dependencies,
            code=code,
            dependencies=deps,
            loaded_files=loaded_files,
            summary=combined_summary,
            plc=plc,
            # tmc_symbols=list(tmc.find(pytmc.parser.Symbol, recurse=False)) if tmc else None,
        )

    @classmethod
    def from_project_filename(
        cls,
        project: AnyPath,
        include_dependencies: bool = True,
        plc_whitelist: Optional[List[str]] = None,
    ) -> Generator[PlcProjectMetadata, None, None]:
        """Given a project/solution filename, get all PlcProjectMetadata."""
        solution = make_solution_from_files(project)
        logger.debug(
            "Solution path %s projects %s", solution.filename, solution.projects
        )
        for tsproj_project in solution.projects:
            logger.debug("Found tsproj %s", tsproj_project.name)
            try:
                parsed_tsproj = tsproj_project.load()
            except Exception:
                logger.exception("Failed to load project %s", tsproj_project.name)
                continue

            for plc_name, plc in parsed_tsproj.plcs_by_name.items():
                if plc_whitelist and plc_name not in plc_whitelist:
                    continue

                logger.debug("Found PLC project %s", plc_name)
                plc_md = cls.from_plcproject(
                    plc,
                    include_dependencies=include_dependencies,
                )
                if plc_md is not None:
                    yield plc_md


def load_projects(
    *projects: AnyPath,
    include_dependencies: bool = True,
    plc_whitelist: Optional[List[str]] = None,
) -> List[PlcProjectMetadata]:
    """Load the given projects by filename."""
    result = []
    for project in projects:
        mds = PlcProjectMetadata.from_project_filename(
            project,
            include_dependencies=include_dependencies,
            plc_whitelist=plc_whitelist,
        )
        result.extend(mds)
    return result
