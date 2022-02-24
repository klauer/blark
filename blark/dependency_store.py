"""
TwinCAT project dependency handling.
"""
from __future__ import annotations

# import collections
import dataclasses
import distutils.version
import functools
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pytmc
import pytmc.code

from . import parse
from . import transform as tf
from . import util
from .config import BLARK_TWINCAT_ROOT
from .summary import CodeSummary

AnyPath = Union[str, pathlib.Path]

logger = logging.getLogger(__name__)
_dependency_store = None


@dataclass
class ResolvedDependency:
    """Resolved dependency version information."""
    name: str
    vendor: str
    version: str
    vendor_short: str


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
        def get_version(path):
            try:
                version = path.name.lstrip('v').replace('-', '.')
                version = tuple(distutils.version.LooseVersion(version).version)
                if isinstance(version[0], int):
                    return version
            except Exception:
                ...

        project_root = root / self.path

        paths = {
            (get_version(path), path) for path in project_root.iterdir()
            if get_version(path) is not None
        }

        for version, path in reversed(sorted(paths)):
            project_fn = path / self.project
            if project_fn.exists():
                logger.debug(
                    "Found latest %s (%s) in %s",
                    self.name, version, project_fn
                )
                return project_fn

        raise FileNotFoundError(
            f"No valid versions of {self.name} found in {project_root}"
        )

    def get_project_filename(self, root: pathlib.Path, version: str) -> pathlib.Path:
        """Get the full project filename, given the root path and version."""
        if not self.versioned:
            return root / self.path / self.project
        if version == "*":
            return self.get_latest_version_path(root) / self.project

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
        try:
            config = self._read_config()
        except FileNotFoundError:
            logger.warning(
                "pytmc dependencies will not be loaded as either "
                "BLARK_TWINCAT_ROOT is unset or invalid.  Expected "
                "file %s to exist",
                self.root / "config.json"
            )
            self.config = DependencyStoreConfig(filename=None, libraries={})
            return

        self.config = DependencyStoreConfig.from_dict(
            config, filename=self.config_filename
        )

    @functools.lru_cache(maxsize=50)
    def get_dependency(self, name: str, version: str) -> List[PlcProjectMetadata]:
        """Get a dependency by name and version number."""
        try:
            info: DependencyStoreLibrary = self.config.libraries[name]
        except KeyError:
            return []

        try:
            filename = info.get_project_filename(self.root, version=version)
        except FileNotFoundError:
            return []

        if not filename.exists():
            return []

        return list(PlcProjectMetadata.from_project_filename(str(filename.resolve())))

    def get_dependencies(
        self, plc: pytmc.parser.Plc,
    ) -> Generator[Tuple[ResolvedDependency, PlcProjectMetadata], None, None]:
        """Get dependency projects from a PLC."""
        for resolution in plc.root.find(pytmc.parser.Resolution):
            resolution: pytmc.parser.Resolution
            try:
                info = ResolvedDependency(**resolution.resolution)
            except (KeyError, ValueError):
                continue

            for proj in self.get_dependency(info.name, info.version):
                yield info, proj

    @staticmethod
    def get_instance() -> DependencyStore:
        """Get the global DependencyStore instance."""
        return get_dependency_store()


def get_dependency_store() -> DependencyStore:
    """Get the global DependencyStore instance."""
    global _dependency_store

    if _dependency_store is None:
        _dependency_store = DependencyStore(
            root=pathlib.Path(BLARK_TWINCAT_ROOT)
        )
    return _dependency_store


@dataclass
class PlcSymbolMetadata:
    context: list
    name: str
    type: str


@dataclass
class PlcProjectMetadata:
    """This is a per-PLC project metadata container."""
    # context: FullLoadContext
    name: str
    filename: str
    include_dependencies: bool
    include_symbols: bool
    code: List[tf.SourceCode]
    summary: CodeSummary
    symbols: Dict[str, PlcSymbolMetadata]
    record_to_symbol: Dict[str, str]
    loaded_files: Dict[str, str]
    dependencies: Dict[str, ResolvedDependency]
    tmc: Optional[pytmc.parser.TcModuleClass]

    # def find_declaration_from_symbol(self, name: str) -> Optional[Declaration]:
    #     """Given a symbol name, find its Declaration."""
    #     parts = collections.deque(name.split("."))
    #     if len(parts) <= 1:
    #         return

    #     variable_name = parts.pop()
    #     parent = None
    #     path = []
    #     while parts:
    #         part = parts.popleft()
    #         if "[" in part:  # ]
    #             part = part.split("[")[0]  # ]

    #         try:
    #             if parent is None:
    #                 parent: PlcCode = self.code[part]
    #             else:
    #                 part_type = parent.declarations[part].type
    #                 parent: PlcCode = self.code[part_type]
    #         except KeyError:
    #             return

    #         path.append(parent)

    #     try:
    #         return parent.declarations[variable_name]
    #     except KeyError:
    #         # Likely ``EXTENDS``, which is not yet supported
    #         ...

    def get_symbol_context(
        self,
        type_name: str,
        symbol_name: str,
    ):  # -> FullLoadContext:
        """Get context information for a given pytmc Symbol."""
        context = []
        try:
            type_context = (
                self.summary.data_types[type_name].name,
                self.summary.data_types[type_name].meta.line
            )
        except KeyError:
            ...
        else:
            context.append(type_context)

        # try:
        #     code, first_symbol, *_ = symbol_name.split(".")
        #     first_context = self.code[code].declarations[first_symbol].context
        # except Exception:
        #     ...
        # else:
        #     context.extend(first_context)

        # decl = self.find_declaration_from_symbol(symbol_name)
        # if decl is not None:
        #     context.extend(decl.context)

        return context  # tuple(remove_redundant_context(context))

    def get_symbol_metadata(
        self,
        symbol: pytmc.parser.Symbol,
        require_records: bool = True
    ) -> Generator[PlcSymbolMetadata, None, None]:
        """Get symbol metadata given a pytmc Symbol."""
        symbol_type_name = symbol.data_type.qualified_type_name
        context = self.get_symbol_context(symbol_type_name, symbol.name)
        if context:
            yield PlcSymbolMetadata(
                context=context,
                name=symbol.name,
                type=symbol_type_name,
            )

    @classmethod
    def from_pytmc(
        cls,
        plc: pytmc.parser.Plc,
        include_dependencies: bool = True,
        include_symbols: bool = True,
        loaded_files: Optional[Dict[str, str]] = None,
    ) -> Optional[PlcProjectMetadata]:
        """Create a PlcProjectMetadata instance from a pytmc-parsed one."""
        tmc = plc.tmc
        if tmc is None and include_symbols:
            logger.debug("%s: No TMC file for symbols; skipping...", plc.name)
            return

        filename = str(plc.filename.resolve())

        loaded_files = dict(loaded_files or {})
        deps = {}
        code = []
        symbols = {}
        combined_summary = CodeSummary()

        loaded_files[filename] = util.get_file_sha256(filename)

        if include_dependencies:
            store = get_dependency_store()
            for resolution, proj in store.get_dependencies(plc):
                code.update(proj.code)
                deps.update(proj.dependencies)
                loaded_files.update(proj.loaded_files)
                deps[resolution.name] = resolution

        for code_path, code_obj in parse.parse_plc(plc, transform=True):
            if isinstance(code_obj, Exception):
                logger.debug("Failed to load: %s %s", code_path, code_obj)
                continue
            code.append(code_obj)
            loaded_files[code_path] = util.get_file_sha256(code_path)
            combined_summary.append(CodeSummary.from_source(code_obj))

        md = cls(
            name=plc.name,
            filename=filename,
            include_dependencies=include_dependencies,
            include_symbols=include_symbols,
            # context=(LoadContext(filename, 0), ),
            code=code,
            symbols=symbols,
            record_to_symbol={},
            dependencies=deps,
            loaded_files=loaded_files,
            summary=combined_summary,
            tmc=tmc,
            # nc=nc,
        )

        if not include_symbols:
            return md

        all_symbols = list(tmc.find(pytmc.parser.Symbol, recurse=False))

        def by_name(symbol):
            return symbol.name

        for symbol in sorted(all_symbols, key=by_name):
            for symbol_md in md.get_symbol_metadata(symbol):
                symbols[symbol_md.name] = symbol_md

        logger.debug(
            "%s: Found %d symbols (%d generated metadata)",
            plc.name, len(all_symbols), len(symbols)
        )

        return md

    @classmethod
    def from_project_filename(
        cls,
        project: AnyPath,
        include_dependencies: bool = True,
        include_symbols: bool = True,
        plc_whitelist: Optional[List[str]] = None,
        loaded_files: Optional[Dict[str, str]] = None,
    ) -> Generator[PlcProjectMetadata, None, None]:
        """Given a project/solution filename, get all PlcProjectMetadata."""
        loaded_files = dict(loaded_files or {})
        solution_path, projects = util.get_tsprojects_from_filename(project)
        logger.debug("Solution path %s projects %s", solution_path, projects)
        for tsproj_project in projects:
            logger.warning("Found tsproj %s", tsproj_project.name)
            try:
                parsed_tsproj = pytmc.parser.parse(tsproj_project)
            except Exception:
                logger.exception("Failed to load project %s", tsproj_project.name)
                continue

            for plc_name, plc in parsed_tsproj.plcs_by_name.items():
                if plc_whitelist and plc_name not in plc_whitelist:
                    continue

                logger.debug("Found plc project %s", plc_name)
                plc_md = cls.from_pytmc(
                    plc,
                    include_dependencies=include_dependencies,
                    include_symbols=include_symbols,
                    loaded_files=loaded_files,
                )
                if plc_md is not None:
                    yield plc_md


if __name__ == "__main__":
    projects = list(
        PlcProjectMetadata.from_project_filename(
            "/Users/klauer/Repos/btps-prototype/btps-prototype.sln"
        )
    )
