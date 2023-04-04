from __future__ import annotations

import codecs
import copy
import dataclasses
import functools
import logging
import pathlib
import re
from collections.abc import Generator
from typing import Any, ClassVar, Optional, Union

import lxml
import lxml.etree

from . import util
from .input import BlarkCompositeSourceItem, BlarkSourceItem
from .typing import ContainsBlarkCode
from .util import AnyPath, SourceType

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

BlarkInputCode = Union[BlarkCompositeSourceItem, BlarkSourceItem]


logger = logging.getLogger(__name__)


_solution_project_regex = re.compile(
    (
        r"^Project\("
        r'"(?P<solution_guid>.*)"'
        r"\)"
        r"\s*=\s*"
        r'"(?P<project_name>.*?)"'
        r"\s*,\s*"
        r'"(?P<project_path>.*?)"'
        r"\s*,\s*"
        r'"(?P<project_guid>.*?)"'
        r"\s*$"
    ),
    re.MULTILINE,
)


@functools.lru_cache(maxsize=2048)
def strip_xml_namespace(tag: str) -> str:
    """Strip off {{namespace}} from: {{namespace}}tag."""
    return lxml.etree.QName(tag).localname


def parse_xml_file(fn: AnyPath) -> lxml.etree.Element:
    """Parse a given XML file with lxml.etree.parse."""
    fn = util.fix_case_insensitive_path(fn)

    with open(fn, "rb") as f:
        tree = lxml.etree.parse(f)

    return tree.getroot()


def parse_xml_contents(contents: str) -> lxml.etree.Element:
    """Parse the given XML contents with lxml.etree."""
    return lxml.etree.fromstring(contents)


def projects_from_solution_source(
    solution_source: str,
    root: pathlib.Path,
) -> Generator[Project, None, None]:
    """
    Find project filenames from the contents of a solution.

    Parameters
    ----------
    solution_source : str
        The solution (.sln) file source.
    """
    for match in _solution_project_regex.finditer(solution_source):
        group = match.groupdict()
        path = pathlib.PureWindowsPath(group["project_path"])
        try:
            local_path = util.fix_case_insensitive_path(root / path)
        except FileNotFoundError:
            local_path = None
            logger.debug(
                "Unable to find local file while loading projects from solution: %s",
                path,
            )

        yield Project(
            name=group["project_name"],
            saved_path=path,
            local_path=local_path,
            guid=group["project_guid"],
            solution_guid=group["solution_guid"],
        )


class SourceCodeFile:
    guid: str
    path: pathlib.Path


def filename_from_xml(xml: Optional[lxml.etree.Element]) -> Optional[pathlib.Path]:
    if xml is None:
        return None

    if hasattr(xml, "getroot"):  # TODO -> switch to isinstance
        root = xml.getroot()
    elif hasattr(xml, "getroottree"):
        root = xml.getroottree()
    else:
        raise ValueError(f"Unexpected type: {type(xml)}")

    if root.docinfo.URL is None:
        return None
    return pathlib.Path(root.docinfo.URL)


def get_child_text(
    xml: lxml.etree.Element,
    tag: str,
    namespace: Optional[str] = None,
    namespaces: Optional[dict[str, str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    namespace = f"{namespace}:" if namespace else ""
    elements = xml.xpath(f"{namespace}{tag}", namespaces=namespaces)
    try:
        return elements[0].text
    except IndexError:
        return default


def get_child_located_text(
    xml: lxml.etree.Element,
    tag: str,
    namespace: Optional[str] = None,
    namespaces: Optional[dict[str, str]] = None,
) -> Optional[LocatedString]:
    namespace = f"{namespace}:" if namespace else ""
    elements = xml.xpath(f"{namespace}{tag}", namespaces=namespaces)
    try:
        return LocatedString(
            line=elements[0].sourceline,
            value=elements[0].text,
        )
    except IndexError:
        return None


@dataclasses.dataclass
class LocatedString:
    line: int
    value: str
    column: int = 0


@dataclasses.dataclass
class TcDeclImpl:
    filename: Optional[pathlib.Path]
    source_type: Optional[SourceType]
    declaration: Optional[LocatedString]
    implementation: Optional[LocatedString]
    metadata: Optional[dict[str, Any]] = None

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.source_type is None:
            return []

        parts = []
        if self.declaration is not None:
            parts.append(
                BlarkSourceItem(
                    file=self.filename,
                    type=self.source_type,
                    implicit_end=None,
                    grammar_rule=self.source_type.get_grammar_rule(),
                    code=self.declaration.value,
                    line=self.declaration.line,
                )
            )

        if self.implementation is not None and self.implementation.value.strip():
            # TODO - not ideal - empty statement list throws error
            if util.remove_all_comments(self.implementation.value).strip():
                parts.append(
                    BlarkSourceItem(
                        file=self.filename,
                        type=self.source_type,
                        implicit_end=None,
                        grammar_rule="statement_list",
                        code=self.implementation.value,
                        line=self.implementation.line,
                    )
                )

        if not parts:
            return []
        # if len(parts) == 1:
        #     return [parts[0]]

        return [
            BlarkCompositeSourceItem(
                type=self.source_type,
                parts=parts,
                implicit_end=self.source_type.get_implicit_block_end(),
                grammar_rule=self.source_type.get_grammar_rule(),
            )
        ]

    def _serialize(self, parent: lxml.etree.Element) -> None:
        if self.declaration is not None:
            decl = lxml.etree.Element("Declaration")
            decl.text = lxml.etree.CDATA(self.declaration.value)
            parent.append(decl)

        if self.implementation is not None:
            impl = lxml.etree.Element("Implementation")
            st = lxml.etree.Element("ST")
            st.text = lxml.etree.CDATA(self.implementation.value)
            impl.append(st)
            parent.append(impl)

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
    ) -> Self:
        declaration = get_child_located_text(xml, "Declaration")
        if declaration is not None:
            source_type = util.find_pou_type(declaration.value)
        else:
            source_type = None
        return cls(
            declaration=declaration,
            # TODO: ST only supported for now
            implementation=get_child_located_text(xml, "Implementation/ST"),
            metadata=xml.attrib,
            source_type=source_type,
            filename=filename,
        )


@dataclasses.dataclass
class TcSource:
    _tag: ClassVar[str] = ""  # TODO: set in subclass
    name: str
    guid: str
    decl: TcDeclImpl
    metadata: dict[str, str]
    filename: Optional[pathlib.Path]

    @property
    def source_type(self) -> Optional[SourceType]:
        if self.decl is None:
            return None
        return self.decl.source_type

    def to_xml(self) -> lxml.etree.Element:
        md = dict(self.metadata)
        plc_obj = lxml.etree.Element("TcPlcObject")
        plc_obj.attrib["Version"] = md.pop("version", "")
        plc_obj.attrib["ProductVersion"] = md.pop("product_version", "")
        primary = lxml.etree.Element(self._tag)
        plc_obj.append(primary)
        self._serialize(primary)
        primary.attrib.update(md)
        return plc_obj

    def _serialize(self, primary: lxml.etree.Element):
        primary.attrib["Name"] = self.name
        primary.attrib["Id"] = self.guid
        if self.decl is not None:
            self.decl._serialize(primary)

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
    ) -> Optional[Union[TcDUT, TcTTO, TcPOU, TcIO, TcGVL]]:
        try:
            tcplc_object = xml.xpath("/TcPlcObject")[0]
        except IndexError:
            return None

        cls_to_xpath = {
            TcDUT: TcDUT._tag,
            TcGVL: TcGVL._tag,
            TcPOU: TcPOU._tag,
            TcIO: TcIO._tag,
            TcTTO: TcTTO._tag,
        }

        for cls, xpath in cls_to_xpath.items():
            items = tcplc_object.xpath(xpath)
            if items:
                item = items[0]
                break
        else:
            return None

        metadata = dict(item.attrib)
        metadata["version"] = tcplc_object.attrib.get("Version", "")
        metadata["product_version"] = tcplc_object.attrib.get("ProductVersion", "")
        decl = TcDeclImpl.from_xml(item, filename=filename)
        kwargs = cls._get_additional_args_from_xml(item, decl, filename=filename)
        return cls(
            filename=filename or filename_from_xml(xml),
            name=metadata.pop("Name", ""),
            guid=metadata.pop("Id", ""),
            decl=decl,
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    def _get_additional_args_from_xml(
        cls,
        xml: lxml.etree.Element,
        decl: Optional[TcDeclImpl],
        *,
        filename: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        return {}

    @classmethod
    def from_contents(
        cls: type[Self],
        contents: str,
        filename: Optional[pathlib.Path] = None,
    ) -> Optional[Union[TcDUT, TcPOU, TcIO, TcTTO, TcGVL]]:
        return cls.from_xml(parse_xml_contents(contents), filename=filename)

    @classmethod
    def from_filename(
        cls: type[Self],
        filename: AnyPath,
    ) -> Optional[Union[TcDUT, TcPOU, TcIO, TcTTO, TcGVL]]:
        with open(filename) as fp:
            raw_contents = fp.read()
        return cls.from_contents(raw_contents, filename=pathlib.Path(filename))


@dataclasses.dataclass
class TcDUT(TcSource):
    _tag: ClassVar[str] = "DUT"
    source_type: ClassVar[SourceType] = SourceType.struct


@dataclasses.dataclass
class TcGVL(TcSource):
    _tag: ClassVar[str] = "GVL"
    source_type: ClassVar[SourceType] = SourceType.var_global


@dataclasses.dataclass
class TcIO(TcSource):
    _tag: ClassVar[str] = "Itf"
    parts: list[Union[TcMethod, TcProperty, TcUnknownXml]]

    def _serialize(self, primary: lxml.etree.Element) -> None:
        super()._serialize(primary)
        for part in self.parts:
            if isinstance(part, (TcUnknownXml, TcExtraInfo)):
                primary.append(copy.deepcopy(part.xml))
            else:
                part_tag = lxml.etree.Element(part._tag)
                part._serialize(part_tag)
                primary.append(part_tag)

    @classmethod
    def create_source_child_from_xml(
        cls,
        child: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
    ) -> Optional[Union[TcAction, TcMethod, TcProperty, TcExtraInfo, TcUnknownXml]]:
        if child.tag in ("Declaration", "Implementation"):
            # Already handled
            return None
        if child.tag == "Action":
            return TcAction.from_xml(child, filename=filename)
        if child.tag == "Method":
            return TcMethod.from_xml(child, filename=filename)
        if child.tag == "Property":
            return TcProperty.from_xml(child, filename=filename)
        if child.tag in ("Folder", "LineIds"):
            return TcExtraInfo.from_xml(child)
        return TcUnknownXml(child)

    @classmethod
    def _get_additional_args_from_xml(
        cls,
        xml: lxml.etree.Element,
        decl: Optional[TcDeclImpl],
        *,
        filename: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        # Only support ST for now:
        parts = [
            cls.create_source_child_from_xml(child, filename=filename)
            for child in xml.iterchildren()
        ]
        return {
            "parts": [part for part in parts if part is not None],
        }


@dataclasses.dataclass
class TcTTO(TcSource):
    _tag: ClassVar[str] = "Task"
    xml: lxml.etree.Element

    def _serialize(self, primary: lxml.etree.Element) -> None:
        primary.getparent().replace(primary, self.xml)

    @classmethod
    def _get_additional_args_from_xml(
        cls,
        xml: lxml.etree.Element,
        decl: Optional[TcDeclImpl],
        *,
        filename: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        return {"xml": xml}


@dataclasses.dataclass
class TcPOU(TcSource):
    _tag: ClassVar[str] = "POU"
    parts: list[Union[TcAction, TcMethod, TcProperty, TcUnknownXml]]

    def to_blark(self) -> list[BlarkInputCode]:
        if self.source_type is None:
            raise RuntimeError("No source type set?")

        items = []

        if self.decl is not None:
            items.extend(self.decl.to_blark())

        for part in self.parts:
            if isinstance(part, ContainsBlarkCode):
                items.extend(part.to_blark())
            elif not isinstance(part, (TcExtraInfo, TcUnknownXml)):
                raise NotImplementedError(
                    f"TcPOU portion {type(part)} not yet implemented"
                )

        return items

    def _serialize(self, primary: lxml.etree.Element) -> None:
        super()._serialize(primary)
        for part in self.parts:
            if isinstance(part, (TcUnknownXml, TcExtraInfo)):
                primary.append(copy.deepcopy(part.xml))
            else:
                part_tag = lxml.etree.Element(part._tag)
                part._serialize(part_tag)
                primary.append(part_tag)

    @classmethod
    def create_source_child_from_xml(
        cls,
        child: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
    ) -> Optional[Union[TcAction, TcMethod, TcProperty, TcExtraInfo, TcUnknownXml]]:
        if child.tag in ("Declaration", "Implementation"):
            # Already handled
            return None
        if child.tag == "Action":
            return TcAction.from_xml(child, filename=filename)
        if child.tag == "Method":
            return TcMethod.from_xml(child, filename=filename)
        if child.tag == "Property":
            return TcProperty.from_xml(child, filename=filename)
        if child.tag in ("Folder", "LineIds"):
            return TcExtraInfo.from_xml(child)
        return TcUnknownXml(child)

    @classmethod
    def _get_additional_args_from_xml(
        cls,
        xml: lxml.etree.Element,
        decl: Optional[TcDeclImpl],
        *,
        filename: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        parts = [
            cls.create_source_child_from_xml(child, filename=filename)
            for child in xml.iterchildren()
        ]
        return {
            "parts": [part for part in parts if part is not None],
        }


@dataclasses.dataclass
class TcSourceChild(TcSource):
    def to_blark(self) -> list[BlarkInputCode]:
        if self.decl.declaration is None:
            return []
        return self.decl.to_blark()

    def _serialize(self, parent: lxml.etree.Element) -> None:
        super()._serialize(parent)
        parent.attrib.update(self.metadata or {})

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
    ) -> Self:
        metadata = dict(xml.attrib)
        decl = TcDeclImpl.from_xml(xml, filename=filename)
        kwargs = cls._get_additional_args_from_xml(xml, decl, filename=filename)
        return cls(
            filename=filename or filename_from_xml(xml),
            name=metadata.pop("Name", ""),
            guid=metadata.pop("Id", ""),
            decl=decl,
            metadata=metadata,
            **kwargs,
        )


@dataclasses.dataclass
class TcMethod(TcSourceChild):
    _tag: ClassVar[str] = "Method"


@dataclasses.dataclass
class TcAction(TcSourceChild):
    _tag: ClassVar[str] = "Action"


@dataclasses.dataclass
class TcProperty(TcSourceChild):
    _tag: ClassVar[str] = "Property"

    get: Optional[TcDeclImpl]
    set: Optional[TcDeclImpl]

    def to_blark(self) -> list[BlarkInputCode]:
        # NOTE: ignoring super().to_blark()
        try:
            base_decl = self.decl.to_blark()[0]
        except IndexError:
            # The top-level Declaration holds the first line - ``PROPERTY name``
            # If it's not there, we don't have a property to use.
            return []

        items = []
        for get_set in (self.get, self.set):
            if get_set is not None:
                (blark_input,) = get_set.to_blark()
                items.append(
                    BlarkCompositeSourceItem(
                        type=SourceType.property,
                        parts=[base_decl, blark_input],
                        grammar_rule=SourceType.property.get_grammar_rule(),
                        implicit_end="END_PROPERTY",
                    )
                )

        return items

    def _serialize(self, parent: lxml.etree.Element) -> None:
        super()._serialize(parent)
        if self.get is not None:
            get = lxml.etree.Element("Get")
            parent.append(get)
            self.get._serialize(get)
            get.attrib.update(self.get.metadata or {})

        if self.set is not None:
            set = lxml.etree.Element("Set")
            parent.append(set)
            self.set._serialize(set)
            set.attrib.update(self.set.metadata or {})

    @classmethod
    def _get_additional_args_from_xml(
        cls,
        xml: lxml.etree.Element,
        decl: Optional[TcDeclImpl],
        *,
        filename: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        try:
            get_section = xml.xpath("Get")[0]
        except IndexError:
            get = None
        else:
            get = TcDeclImpl.from_xml(get_section, filename=filename)
            get.source_type = SourceType.property_get

        try:
            set_section = xml.xpath("Set")[0]
        except IndexError:
            set = None
        else:
            set = TcDeclImpl.from_xml(set_section, filename=filename)
            set.source_type = SourceType.property_set
        return dict(get=get, set=set)


@dataclasses.dataclass
class TcExtraInfo:
    metadata: dict[str, str]
    xml: lxml.etree.Element

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
    ) -> Self:
        return cls(metadata=xml.attrib, xml=xml)


@dataclasses.dataclass
class TcUnknownXml:
    xml: lxml.etree.Element


@dataclasses.dataclass
class TwincatSourceCodeItem:
    saved_path: pathlib.PureWindowsPath
    local_path: Optional[pathlib.Path]
    subtype: Optional[str]
    link_always: bool
    guid: Optional[str] = None
    raw_contents: Optional[str] = None
    contents: Optional[Union[TcDUT, TcPOU, TcIO, TcGVL, TcTTO]] = None

    def to_string(self, delimiter: str = "\r\n") -> str:
        if self.contents is None:
            raise ValueError(
                f"No contents to save (file not found on host for {self.saved_path})"
            )

        lines = ['<?xml version="1.0" encoding="utf-8"?>']
        tree = self.contents.to_xml()
        lxml.etree.indent(tree, space=" " * 2)
        lines += (
            lxml.etree.tostring(
                tree,
                pretty_print=True,
                encoding="utf-8",
            )
            .decode("utf-8")
            .splitlines()
        )
        return delimiter.join(lines)

    def save_to(self, path: AnyPath, delimiter: str = "\r\n") -> None:
        code = self.to_string(delimiter)
        with codecs.open(str(path), "w", "utf-8-sig") as fp:
            fp.write(code)

    @classmethod
    def from_compile_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
    ) -> Optional[Self]:
        saved_path = pathlib.PureWindowsPath(xml.attrib["Include"])
        try:
            plcproj_filename = filename_from_xml(xml)
            if plcproj_filename is None:
                raise FileNotFoundError("No project filename?")
            local_path = util.fix_case_insensitive_path(
                plcproj_filename.parent / saved_path
            )
        except FileNotFoundError:
            local_path = None
            raw_contents = None
            contents = None
            logger.debug(
                "Unable to find local file while loading projects from solution: %s",
                saved_path,
                exc_info=True,
            )
        else:
            with open(local_path) as fp:
                raw_contents = fp.read()
            contents = TcSource.from_contents(raw_contents, filename=local_path)

        namespaces = {"msbuild": xml.xpath("namespace-uri()")}
        subtype = get_child_text(
            xml,
            "SubType",
            namespace="msbuild",
            namespaces=namespaces,
            default="",
        )
        link_always = (
            get_child_text(
                xml,
                "LinkAlways",
                namespace="msbuild",
                namespaces=namespaces,
                default="false",
            )
            == "true"
        )

        return cls(
            contents=contents,
            guid=None,
            link_always=link_always,
            local_path=local_path,
            raw_contents=raw_contents,
            saved_path=saved_path,
            subtype=subtype,
        )


@dataclasses.dataclass
class TwincatPlcProject:
    guid: str
    xti_path: Optional[pathlib.Path]
    plcproj_path: Optional[pathlib.Path]
    properties: dict[str, str]
    sources: list[TwincatSourceCodeItem]

    @classmethod
    def from_standalone_xml(
        cls: type[Self],
        xti_xml: Optional[lxml.etree.Element],
        plcproj_xml: lxml.etree.Element,
    ) -> Self:
        namespaces = {"msbuild": plcproj_xml.xpath("namespace-uri()")}
        properties = {
            strip_xml_namespace(element.tag): element.text
            for element in plcproj_xml.xpath(
                "/msbuild:Project/msbuild:PropertyGroup/msbuild:*",
                namespaces=namespaces,
            )
        }
        sources = [
            TwincatSourceCodeItem.from_compile_xml(xml)
            for xml in plcproj_xml.xpath(
                "/msbuild:Project/msbuild:ItemGroup/msbuild:Compile",
                namespaces=namespaces,
            )
        ]
        return cls(
            guid=properties.get("ProjectGuid", ""),
            xti_path=filename_from_xml(xti_xml),
            plcproj_path=filename_from_xml(plcproj_xml),
            sources=[source for source in sources if source is not None],
            properties=properties,
        )

    @classmethod
    def from_xti_filename(cls: type[Self], xti_filename: pathlib.Path) -> Self:
        xti = parse_xml_file(xti_filename)
        (project,) = xti.xpath("/TcSmItem/Project")
        prj_file_path = pathlib.PureWindowsPath(project.attrib.get("PrjFilePath"))

        plcproj_path = util.fix_case_insensitive_path(
            xti_filename.parent / prj_file_path
        )
        plcproj = parse_xml_file(plcproj_path)
        return cls.from_standalone_xml(xti, plcproj)

    @classmethod
    def from_project_xml(
        cls: type[Self],
        plc_project: lxml.etree.Element,
        root: pathlib.Path,
    ) -> Self:
        # tsproj -> xti -> plcproj
        file = plc_project.attrib.get("File", None)
        if file is None:
            # return cls.from_standalone_xml(plc_project, None)
            raise NotImplementedError

        plc_filename = pathlib.Path(file)
        xti_filename = root / "_Config" / "PLC" / plc_filename
        return cls.from_xti_filename(xti_filename=xti_filename)

    # @classmethod
    # def from_filename(cls: type[Self], filename: pathlib.Path) -> Self:
    #     return cls.from_standalone_xml(parse_xml_file(filename), filename.parent)


def get_project_guid(element: lxml.etree.Element) -> str:
    try:
        proj = element.xpath("/TcSmProject/Project")[0]
    except IndexError:
        return ""

    return proj.attrib.get("ProjectGUID", "")


def get_project_target_netid(element: lxml.etree.Element) -> str:
    try:
        proj = element.xpath("/TcSmProject/Project")[0]
    except IndexError:
        return ""

    return proj.attrib.get("TargetNetId", "")


@dataclasses.dataclass
class TwincatTsProject:
    guid: str
    netid: str
    path: pathlib.Path
    plcs: list[TwincatPlcProject]

    @classmethod
    def from_filename(cls, filename: pathlib.Path) -> TwincatTsProject:
        tsproj = parse_xml_file(filename)
        plcs = [
            TwincatPlcProject.from_project_xml(plc_project, filename.parent)
            for plc_project in tsproj.xpath("/TcSmProject/Project/Plc/Project")
        ]

        return TwincatTsProject(
            path=filename,
            guid=get_project_guid(tsproj),
            netid=get_project_target_netid(tsproj),
            plcs=plcs,
        )


@dataclasses.dataclass
class Project:
    name: str
    saved_path: pathlib.PureWindowsPath
    local_path: Optional[pathlib.Path]
    guid: str
    solution_guid: str

    # def find_source_code(
    #     self, root: pathlib.Path
    # ) -> Generator[SourceCodeFile, None, None]:
    #     ...
    #
    def load(self):
        if self.local_path is None:
            raise FileNotFoundError(
                f"File from project settings not found: {self.saved_path}"
            )

        if self.local_path.suffix.lower() in (".tsproj",):
            return TwincatTsProject.from_filename(self.local_path)

        raise NotImplementedError(f"Format not yet supported: {self.local_path.suffix}")


@dataclasses.dataclass
class Solution:
    root: pathlib.Path
    projects: list[Project]
    filename: Optional[pathlib.Path] = None

    @property
    def projects_by_name(self) -> dict[str, Project]:
        return {project.name: project for project in self.projects}

    @classmethod
    def from_contents(
        cls: type[Self],
        solution_source: str,
        root: pathlib.Path,
        filename: Optional[pathlib.Path] = None,
    ) -> Self:
        return cls(
            root=root,
            filename=filename,
            projects=list(projects_from_solution_source(solution_source, root=root)),
        )

    @classmethod
    def from_filename(cls: type[Self], filename: AnyPath) -> Self:
        filename = pathlib.Path(filename).expanduser().resolve()
        with open(filename, "rt") as f:
            solution_source = f.read()
        return cls.from_contents(
            solution_source=solution_source,
            root=filename.parent,
            filename=filename,
        )
