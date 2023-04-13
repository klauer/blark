"""
``blark.solution`` - TwinCAT solution/tsproj/plcproj/source code loading helpers.


TwinCAT source file extensions for reference:
    .tcdut    data unit type
    .tcgtlo   global text list object
    .tcgvl    global variable list
    .tcio     interface
    .tcipo    image pool
    .tcpou    program organization unit
    .tcrmo    recipe manager
    .tctlo    text list object
    .tctto    task object
    .tcvis    visualization
    .tcvmo    visualization manager object
    .tmc      module class - description of project
    .tpy      tmc-like inter-vendor format
    .xti      independent project file
    .sln      Visual Studio solution
    .tsproj   TwinCAT project
    .plcproj  TwinCAT PLC project
"""
from __future__ import annotations

import codecs
import copy
import dataclasses
import functools
import logging
import pathlib
import re
from collections.abc import Generator
from typing import Any, ClassVar, List, Optional, Union

import lxml
import lxml.etree

from . import util
from .input import (BlarkCompositeSourceItem, BlarkSourceItem, BlarkSourceLine,
                    register_file_handler)
from .typing import ContainsBlarkCode, Self
from .util import AnyPath, SourceType

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


def strip_implicit_lines(code: str, source_type: SourceType) -> str:
    """Strip off (e.g.) END_FUNCTION_BLOCK the provided code."""
    implicit_end = source_type.get_implicit_block_end()
    if not implicit_end or not code:
        return code

    # Don't think too hard about this; blark appends this consistently.
    # We shouldn't have to do more than this, I think.  Reformatting tools
    # should be careful around the implicit lines.
    lines = list(code.splitlines())
    # NOTE: switched actions to statement_list
    # if source_type == SourceType.action:
    #     if lines[0].startswith("ACTION "):
    #         lines.pop(0)

    if lines[-1] == implicit_end:
        lines.pop(-1)
    return "\n".join(lines)


def parse_xml_file(fn: AnyPath) -> lxml.etree.Element:
    """Parse a given XML file with lxml.etree.parse."""
    fn = util.fix_case_insensitive_path(fn)

    with open(fn, "rb") as f:
        tree = lxml.etree.parse(f)

    return tree.getroot()


def parse_xml_contents(contents: Union[bytes, str]) -> lxml.etree.Element:
    """Parse the given XML contents with lxml.etree."""
    if isinstance(contents, str):
        contents = contents.encode("utf-8")
    return lxml.etree.fromstring(contents)


def projects_from_solution_source(
    solution_source: Union[str, bytes],
    root: pathlib.Path,
    encoding: str = "utf-8",
) -> Generator[Project, None, None]:
    """
    Find project filenames from the contents of a solution.

    Parameters
    ----------
    solution_source : str
        The solution (.sln) file source.
    root : pathlib.Path
        The root path to look for project files.
    """
    if isinstance(solution_source, bytes):
        solution_source = solution_source.decode(encoding)

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
    filename: Optional[pathlib.Path] = None,
) -> Optional[LocatedString]:
    namespace = f"{namespace}:" if namespace else ""
    elements = xml.xpath(f"{namespace}{tag}", namespaces=namespaces)
    try:
        return LocatedString(
            filename=filename,
            lineno=elements[0].sourceline,
            value=elements[0].text,
        )
    except IndexError:
        return None


@dataclasses.dataclass
class Identifier:
    parts: List[str]
    decl_impl: Optional[str]

    def to_string(self) -> str:
        parts = ".".join(self.parts)
        if self.decl_impl:
            return f"{parts}/{self.decl_impl}"
        return parts

    @classmethod
    def from_string(cls: type[Self], value: str) -> Self:
        if "/" in value:
            identifier, decl_impl = value.split("/")
            return cls(
                parts=identifier.split("."),
                decl_impl=decl_impl,
            )
        return cls(
            parts=value.split("."),
            decl_impl=None,
        )


@dataclasses.dataclass
class LocatedString:
    filename: Optional[pathlib.Path]
    lineno: int = 0
    value: str = ""
    column: int = 0

    def to_lines(self) -> list[BlarkSourceLine]:
        return BlarkSourceLine.from_code(
            self.value,
            first_lineno=self.lineno,
            filename=self.filename,
        )


@dataclasses.dataclass
class TcDeclImpl:
    identifier: str
    filename: Optional[pathlib.Path]
    source_type: Optional[SourceType]
    declaration: Optional[LocatedString]
    implementation: Optional[LocatedString]
    parent: Optional[TcSource] = None
    metadata: Optional[dict[str, Any]] = None

    def rewrite_code(self, identifier: str, contents: str):
        if self.source_type is not None:
            contents = strip_implicit_lines(contents, self.source_type)

        ident = Identifier.from_string(identifier)
        if ident.decl_impl == "declaration":
            if self.declaration is None:
                self.declaration = LocatedString(filename=self.filename)
            self.declaration.value = contents
        elif ident.decl_impl == "implementation":
            if self.implementation is None:
                self.implementation = LocatedString(filename=self.filename)
            self.implementation.value = contents
        else:
            raise ValueError(
                f"Unexpected rewrite portion: {identifier} ({ident.decl_impl})"
            )

    def to_blark(self) -> list[BlarkSourceItem]:
        if self.source_type is None:
            return []

        res = []
        if self.declaration is not None:
            decl = BlarkSourceItem(
                identifier=f"{self.identifier}/declaration",
                type=self.source_type,
                lines=self.declaration.to_lines(),
                grammar_rule=self.source_type.get_grammar_rule(),
                implicit_end=self.source_type.get_implicit_block_end(),
                user=self.parent or self,
            )
            res.append(decl)

        if self.implementation is not None:
            # TODO: statement_list in the grammar cannot be empty.  We
            # pre-filter the implementation to ensure that it has some code.
            # If this affects how you're using blark, feel free to open an issue
            # and we can resolve it.
            if util.remove_all_comments(self.implementation.value).strip():
                impl = BlarkSourceItem(
                    identifier=f"{self.identifier}/implementation",
                    type=self.source_type,
                    lines=self.implementation.to_lines(),
                    grammar_rule=SourceType.statement_list.name,
                    implicit_end=SourceType.statement_list.get_implicit_block_end(),
                    user=self.parent or self,
                )
                res.append(impl)

        return res

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
        declaration = get_child_located_text(xml, "Declaration", filename=filename)
        if declaration is not None:
            source_type, identifier = util.find_pou_type_and_identifier(
                declaration.value
            )
        else:
            source_type, identifier = None, None
        return cls(
            identifier=identifier or (filename and filename.stem) or "unknown",
            declaration=declaration,
            # TODO: ST only supported for now
            implementation=get_child_located_text(
                xml, "Implementation/ST", filename=filename
            ),
            metadata=xml.attrib,
            source_type=source_type,
            filename=filename,
        )


def get_tcplc_from_xml(
    xml: lxml.etree.Element,
) -> Optional[lxml.etree.Element]:
    try:
        return xml.xpath("/TcPlcObject")[0]
    except IndexError:
        return None


def get_code_object_from_xml(
    xml: lxml.etree.Element,
) -> tuple[Optional[type[TcSource]], Optional[lxml.etree.Element]]:
    tcplc_object = get_tcplc_from_xml(xml)
    if tcplc_object is not None:
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
                return cls, items[0]

    return None, None


@dataclasses.dataclass
class TcSource:
    _tag: ClassVar[str] = ""  # TODO: set in subclass
    name: str
    guid: str
    decl: TcDeclImpl
    metadata: dict[str, str]
    filename: Optional[pathlib.Path]
    parent: Optional[TwincatSourceCodeItem]
    xml: dataclasses.InitVar[lxml.etree.Element | None] = None

    @property
    def source_type(self) -> Optional[SourceType]:
        if self.decl is None:
            return None
        return self.decl.source_type

    def rewrite_code(self, identifier: str, contents: str):
        raise NotImplementedError(
            f"Rewriting code not yet supported for {type(self)}"
        )

    def to_file_contents(self) -> str:
        parent_to_file_contents = getattr(self.parent, "to_file_contents", None)
        if parent_to_file_contents is not None:
            # If we have a parent, we can't serialize just part of the file.
            # Serialize the whole thing.
            return parent_to_file_contents()

        tree = self.to_xml()
        lxml.etree.indent(tree, space=" " * 2)
        return lxml.etree.tostring(tree)

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

    @staticmethod
    def from_xml(
        xml: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
        parent: Optional[TcSource] = None,
    ) -> Optional[Union[TcDUT, TcTTO, TcPOU, TcIO, TcGVL]]:
        tcplc_object = get_tcplc_from_xml(xml)
        if tcplc_object is None:
            return None
        source_cls, item = get_code_object_from_xml(xml)
        if source_cls is None or item is None:
            raise RuntimeError(
                f"Unsupported xml type for TcSource: {xml}"
            )

        metadata = dict(item.attrib)
        metadata["version"] = tcplc_object.attrib.get("Version", "")
        metadata["product_version"] = tcplc_object.attrib.get("ProductVersion", "")

        decl = TcDeclImpl.from_xml(item, filename=filename)

        source = source_cls(
            filename=filename or filename_from_xml(xml),
            name=metadata.pop("Name", ""),
            guid=metadata.pop("Id", ""),
            decl=decl,
            metadata=metadata,
            parent=parent,
            xml=xml,
        )
        source.decl.parent = source
        return source

    @classmethod
    def from_contents(
        cls: type[Self],
        contents: bytes,
        filename: Optional[pathlib.Path] = None,
        parent: Optional[TcSource] = None,
    ) -> Optional[Union[TcDUT, TcPOU, TcIO, TcTTO, TcGVL]]:
        return cls.from_xml(parse_xml_contents(contents), filename=filename, parent=parent)

    @classmethod
    def from_filename(
        cls: type[Self],
        filename: AnyPath,
        parent: Optional[TcSource] = None,
    ) -> Optional[Union[TcDUT, TcPOU, TcIO, TcTTO, TcGVL]]:
        with open(filename, "rb") as fp:
            raw_contents = fp.read()
        return cls.from_contents(raw_contents, filename=pathlib.Path(filename), parent=parent)


@dataclasses.dataclass
class TcDUT(TcSource):
    _tag: ClassVar[str] = "DUT"
    file_extension: ClassVar[str] = ".TcDUT"
    source_type: ClassVar[SourceType] = SourceType.struct

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.decl is not None:
            res = self.decl.to_blark()
            for item in res:
                item.user = self
            return res
        return []


@dataclasses.dataclass
class TcGVL(TcSource):
    _tag: ClassVar[str] = "GVL"
    file_extension: ClassVar[str] = ".TcGVL"
    source_type: ClassVar[SourceType] = SourceType.var_global

    def rewrite_code(self, identifier: str, contents: str):
        # TODO: need to not save the implicit end line
        return self.decl.rewrite_code(identifier, contents)

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.decl is not None:
            res = self.decl.to_blark()
            for item in res:
                item.user = self
            return res
        return []


@dataclasses.dataclass
class TcIO(TcSource):
    _tag: ClassVar[str] = "Itf"
    file_extension: ClassVar[str] = ".TcIO"

    parts: list[Union[TcMethod, TcProperty, TcUnknownXml]] = dataclasses.field(
        default_factory=list
    )

    def __post_init__(self, xml: Optional[lxml.etree.Element]) -> None:
        if xml is None:
            return
        _, item = get_code_object_from_xml(xml)
        self.parts = [
            self.create_source_child_from_xml(
                child,
                filename=self.filename,
                parent=self,
            )
            for child in item.iterchildren()
            if child.tag not in ("Declaration", "Implementation")
        ]

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.decl is not None:
            res = self.decl.to_blark()
            for item in res:
                item.user = self
            return res
        return []

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
        parent: TcSource,
        filename: Optional[pathlib.Path] = None,
    ) -> Union[TcAction, TcMethod, TcProperty, TcExtraInfo, TcUnknownXml]:
        if child.tag == "Action":
            return TcAction.from_xml(child, filename=filename, parent=parent)
        if child.tag == "Method":
            return TcMethod.from_xml(child, filename=filename, parent=parent)
        if child.tag == "Property":
            return TcProperty.from_xml(child, filename=filename, parent=parent)
        if child.tag in ("Folder", "LineIds"):
            return TcExtraInfo.from_xml(child, parent=parent)
        return TcUnknownXml(child, parent=parent)


@dataclasses.dataclass
class TcTTO(TcSource):
    _tag: ClassVar[str] = "Task"
    file_extension: ClassVar[str] = ".TcTTO"
    xml: Optional[lxml.etree.Element] = None

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        return []

    def _serialize(self, primary: lxml.etree.Element) -> None:
        primary.getparent().replace(primary, self.xml)


@dataclasses.dataclass
class TcPOU(TcSource):
    _tag: ClassVar[str] = "POU"
    file_extension: ClassVar[str] = ".TcPOU"
    parts: list[
        Union[TcAction, TcMethod, TcProperty, TcUnknownXml]
    ] = dataclasses.field(default_factory=list)

    def __post_init__(self, xml: Optional[lxml.etree.Element]) -> None:
        if xml is None:
            return

        _, item = get_code_object_from_xml(xml)
        self.parts = [
            self.create_source_child_from_xml(
                child,
                filename=self.filename,
                parent=self,
            )
            for child in item.iterchildren()
            if child.tag not in ("Declaration", "Implementation")
        ]

    def rewrite_code(self, identifier: str, contents: str):
        # TODO: need to not save the implicit end line
        return self.decl.rewrite_code(identifier, contents)

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.source_type is None:
            raise RuntimeError("No source type set?")

        parts = []

        if self.decl is not None:
            identifier = self.decl.identifier
            parts.extend(self.decl.to_blark())
        else:
            identifier = None

        for part in self.parts:
            if isinstance(part, ContainsBlarkCode):
                for item in part.to_blark():
                    if identifier:
                        item.identifier = f"{identifier}.{item.identifier}"
                    item.user = part
                    parts.append(item)
            elif not isinstance(part, (TcExtraInfo, TcUnknownXml)):
                raise NotImplementedError(
                    f"TcPOU portion {type(part)} not yet implemented"
                )

        return [
            BlarkCompositeSourceItem(
                filename=self.filename,
                identifier=self.decl.identifier if self.decl is not None else "unknown",
                parts=parts,
                user=self,
            )
        ]

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
        parent: TcSource,
        filename: Optional[pathlib.Path] = None,
    ) -> Optional[Union[TcAction, TcMethod, TcProperty, TcExtraInfo, TcUnknownXml]]:
        if child.tag == "Action":
            return TcAction.from_xml(child, filename=filename, parent=parent)
        if child.tag == "Method":
            return TcMethod.from_xml(child, filename=filename, parent=parent)
        if child.tag == "Property":
            return TcProperty.from_xml(child, filename=filename, parent=parent)
        if child.tag in ("Folder", "LineIds"):
            return TcExtraInfo.from_xml(child, parent=parent)
        return TcUnknownXml(child, parent=parent)


@dataclasses.dataclass
class TcSourceChild(TcSource):
    parent: Optional[TcSource] = None

    def _serialize(self, parent: lxml.etree.Element) -> None:
        super()._serialize(parent)
        parent.attrib.update(self.metadata or {})

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
        filename: Optional[pathlib.Path] = None,
        parent: Optional[TcSource] = None,
    ) -> Self:
        metadata = dict(xml.attrib)
        decl = TcDeclImpl.from_xml(xml, filename=filename)
        source = cls(
            filename=filename or filename_from_xml(xml),
            name=metadata.pop("Name", ""),
            guid=metadata.pop("Id", ""),
            decl=decl,
            metadata=metadata,
            parent=parent,
            xml=xml,
        )
        source.decl.parent = source
        return source


@dataclasses.dataclass
class TcMethod(TcSourceChild):
    _tag: ClassVar[str] = "Method"

    def rewrite_code(self, identifier: str, contents: str):
        return self.decl.rewrite_code(identifier, contents)

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.decl.declaration is None:
            return []
        res = self.decl.to_blark()
        for item in res:
            # item.identifier = f"{self.name}/{item.identifier}"
            item.user = self
        return res


@dataclasses.dataclass
class TcAction(TcSourceChild):
    _tag: ClassVar[str] = "Action"

    def rewrite_code(self, identifier: str, contents: str):
        # TODO: need to not save the implicit end line
        # self.decl.implementation.value = contents
        ident = Identifier.from_string(identifier)
        ident.decl_impl = "implementation"
        return self.decl.rewrite_code(ident.to_string(), contents)

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        if self.decl is None or self.decl.implementation is None:
            return []

        if not util.remove_all_comments(self.decl.implementation.value).strip():
            return []

        # Actions have no declaration section; the implementation just has
        # a statement list
        lines = self.decl.implementation.to_lines()
        return [
            BlarkSourceItem(
                identifier=self.name,
                lines=lines,
                type=SourceType.action,
                grammar_rule=SourceType.action.get_grammar_rule(),
                implicit_end=SourceType.action.get_implicit_block_end(),
                user=self,
            )
        ]


@dataclasses.dataclass
class TcProperty(TcSourceChild):
    _tag: ClassVar[str] = "Property"

    get: Optional[TcDeclImpl] = None
    set: Optional[TcDeclImpl] = None

    def __post_init__(self, xml: Optional[lxml.etree.Element]) -> None:
        if xml is None:
            return

        try:
            get_section = xml.xpath("Get")[0]
        except IndexError:
            get = None
        else:
            get = TcDeclImpl.from_xml(get_section, filename=self.filename)
            get.source_type = SourceType.property_get

        try:
            set_section = xml.xpath("Set")[0]
        except IndexError:
            set = None
        else:
            set = TcDeclImpl.from_xml(set_section, filename=self.filename)
            set.source_type = SourceType.property_set

        self.get = get
        self.set = set

    def rewrite_code(self, identifier: str, contents: str):
        # TODO: need to not save the implicit end line
        ident = Identifier.from_string(identifier)
        get_or_set: Optional[TcDeclImpl] = getattr(self, ident.parts[-1])
        if get_or_set is None:
            # get_or_set = TcDeclImpl(identifier=identifier)
            # setattr(self, get_or_set, ident.parts[-1], get_or_set)
            raise NotImplementedError("Empty get/set (TODO)")
        return get_or_set.rewrite_code(identifier, contents)

    def to_blark(self) -> list[Union[BlarkCompositeSourceItem, BlarkSourceItem]]:
        try:
            base_decl: BlarkSourceItem = self.decl.to_blark()[0]
        except IndexError:
            # The top-level Declaration holds the first line - ``PROPERTY name``
            # If it's not there, we don't have a property to use.
            return []

        property_ident = Identifier.from_string(base_decl.identifier)

        parts = []
        for get_or_set, obj in (("get", self.get), ("set", self.set)):
            if obj is None:
                continue

            for part in obj.to_blark():
                part_ident = Identifier.from_string(part.identifier)
                # Note: the parent will add on the FB name
                parts.append(
                    BlarkSourceItem(
                        identifier=Identifier(
                            parts=[*property_ident.parts, get_or_set],
                            decl_impl=part_ident.decl_impl
                        ).to_string(),
                        type=SourceType.property,
                        lines=base_decl.lines + part.lines,
                        grammar_rule=SourceType.property.get_grammar_rule(),
                        implicit_end="END_PROPERTY",
                        user=self,
                    )
                )

        return parts

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


@dataclasses.dataclass
class TcExtraInfo:
    metadata: dict[str, str]
    xml: lxml.etree.Element
    parent: TcSource

    @classmethod
    def from_xml(
        cls: type[Self],
        xml: lxml.etree.Element,
        parent: TcSource,
    ) -> Self:
        return cls(metadata=xml.attrib, xml=xml, parent=parent)


@dataclasses.dataclass
class TcUnknownXml:
    xml: lxml.etree.Element
    parent: TcSource


@dataclasses.dataclass
class TwincatSourceCodeItem:
    saved_path: pathlib.PurePath
    local_path: Optional[pathlib.Path]
    subtype: Optional[str]
    link_always: bool
    guid: Optional[str] = None
    raw_contents: bytes = b''
    contents: Optional[Union[TcDUT, TcPOU, TcIO, TcGVL, TcTTO]] = None
    parent: Optional[TwincatPlcProject] = None

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
        parent: Optional[TwincatPlcProject] = None,
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
            raw_contents = b''
            contents = None
            logger.debug(
                "Unable to find local file while loading projects from solution: %s",
                saved_path,
                exc_info=True,
            )
        else:
            with open(local_path, "rb") as fp:
                raw_contents = fp.read()
            contents = TcSource.from_contents(
                raw_contents,
                filename=local_path,
                parent=None,
            )

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

        item = cls(
            contents=contents,
            guid=None,
            link_always=link_always,
            local_path=local_path,
            raw_contents=raw_contents,
            saved_path=saved_path,
            subtype=subtype,
            parent=parent,
        )
        if contents is not None:
            contents.parent = item
        return item


@dataclasses.dataclass
class DependencyVersion:
    name: str
    version: str
    vendor: str
    namespace: str

    @classmethod
    def from_string(
        cls: type[Self],
        text: str,
        namespace: str,
    ) -> Self:
        library_name, version_and_vendor = text.split(",", 1)
        version, vendor = version_and_vendor.strip().split("(", 1)
        vendor = vendor.rstrip(")")
        version = version.strip()
        return cls(
            name=library_name,
            version=version,
            vendor=vendor,
            namespace=namespace,
        )


@dataclasses.dataclass
class DependencyInformation:
    name: str
    default: Optional[DependencyVersion] = None
    resolution: Optional[DependencyVersion] = None

    @classmethod
    def from_xml(
        cls: type[Self],
        references: list[lxml.etree.Element],
        resolutions: list[lxml.etree.Element],
        xmlns: Optional[dict[str, str]] = None,
    ) -> dict[str, Self]:
        by_name = {}
        for ref in references:
            try:
                name = ref.attrib["Include"]
                res = ref.xpath("msbuild:DefaultResolution", namespaces=xmlns)[0].text
                namespace = ref.xpath("msbuild:Namespace", namespaces=xmlns)[0].text
            except (KeyError, IndexError):
                logger.warning(
                    "Incompatible dependency reference? %s", lxml.etree.tostring(ref)
                )
                continue
            by_name[name] = cls(
                name=name,
                default=DependencyVersion.from_string(res, namespace=namespace),
                resolution=None,
            )

        for ref in resolutions:
            try:
                name = ref.attrib["Include"]
                res = ref.xpath("msbuild:Resolution", namespaces=xmlns)[0].text
            except (KeyError, IndexError):
                logger.warning(
                    "Incompatible dependency reference? %s", lxml.etree.tostring(ref)
                )
                continue
            try:
                namespace = by_name[name].default.namespace
            except KeyError:
                # Unclear if we can infer the namespace without a default here.
                # Let's default to just its name.
                by_name[name] = DependencyInformation(name=name)
                namespace = name

            by_name[name].resolution = DependencyVersion.from_string(
                res, namespace=namespace
            )
        return by_name


@dataclasses.dataclass
class TwincatPlcProject:
    """
    A TwinCAT PLC project.

    This typically corresponds to a single ``.plcproj`` file.
    """

    file_extension: ClassVar[str] = ".plcproj"
    guid: str
    xti_path: Optional[pathlib.Path]
    plcproj_path: Optional[pathlib.Path]
    properties: dict[str, str]
    dependencies: dict[str, DependencyInformation]
    sources: list[TwincatSourceCodeItem]

    @classmethod
    def from_standalone_xml(
        cls: type[Self],
        tsproj_or_xti_xml: Optional[lxml.etree.Element],
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
        dependencies = DependencyInformation.from_xml(
            plcproj_xml.xpath(
                "/msbuild:Project/msbuild:ItemGroup/msbuild:PlaceholderReference",
                namespaces=namespaces,
            ),
            plcproj_xml.xpath(
                "/msbuild:Project/msbuild:ItemGroup/msbuild:PlaceholderResolution",
                namespaces=namespaces,
            ),
            xmlns=namespaces,
        )
        plc_project = cls(
            guid=properties.get("ProjectGuid", ""),
            xti_path=filename_from_xml(tsproj_or_xti_xml),
            plcproj_path=filename_from_xml(plcproj_xml),
            sources=[],
            properties=properties,
            dependencies=dependencies,
        )
        sources = [
            TwincatSourceCodeItem.from_compile_xml(xml, parent=plc_project)
            for xml in plcproj_xml.xpath(
                "/msbuild:Project/msbuild:ItemGroup/msbuild:Compile",
                namespaces=namespaces,
            )
        ]
        plc_project.sources = [source for source in sources if source is not None]
        return plc_project

    @property
    def name(self) -> Optional[str]:
        return self.properties.get("Name", None)

    @classmethod
    def from_filename(
        cls: type[Self],
        filename: pathlib.Path,
    ) -> Self:
        plcproj = parse_xml_file(filename)
        return cls.from_standalone_xml(None, plcproj)

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
        xml: lxml.etree.Element,
        root: pathlib.Path,
    ) -> Self:
        # tsproj -> xti -> plcproj
        file = xml.attrib.get("File", None)
        if file is not None:
            # The PLC project is saved externally in ``xti_filename``.
            plc_filename = pathlib.Path(file)
            xti_filename = root / "_Config" / "PLC" / plc_filename
            return cls.from_xti_filename(xti_filename=xti_filename)

        project_file_path = xml.attrib.get("PrjFilePath")
        if project_file_path is None:
            raise RuntimeError(
                f"Unsupported project saving format; neither 'File' nor 'PrjFilePath' "
                f"is present in the XML attributes of {xml.tag}"
            )

        # The PLC project settings are partly saved in the tsproj file; the rest
        # will come from the .plcproj (as part of PlcProjPath).
        xml_filename = filename_from_xml(xml)
        plcproj_parent = xml_filename.parent if xml_filename is not None else root
        plcproj_path = util.fix_case_insensitive_path(
            plcproj_parent / pathlib.PureWindowsPath(project_file_path)
        )
        plcproj = parse_xml_file(plcproj_path)
        return cls.from_standalone_xml(xml, plcproj)


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
    file_extension: ClassVar[str] = ".tsproj"
    guid: str
    netid: str
    path: Optional[pathlib.Path]
    plcs: list[TwincatPlcProject]

    @property
    def plcs_by_name(self) -> dict[str, TwincatPlcProject]:
        return {plc.name: plc for plc in self.plcs if plc.name is not None}

    @classmethod
    def from_filename(cls, filename: pathlib.Path) -> TwincatTsProject:
        tsproj = parse_xml_file(filename)
        plcs = [
            TwincatPlcProject.from_project_xml(plc_project, root=filename.parent)
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
    saved_path: pathlib.PurePath
    local_path: Optional[pathlib.Path]
    guid: str
    solution_guid: str
    loaded: Optional[TwincatTsProject] = None

    @classmethod
    def from_filename(cls: type[Self], filename: AnyPath) -> Self:
        filename = pathlib.Path(filename)
        if filename.suffix.lower() in (".plcproj",):
            plc = TwincatPlcProject.from_filename(filename)
            loaded = TwincatTsProject(
                guid="",
                netid="",
                path=None,
                plcs=[plc],
            )
        else:
            loaded = TwincatTsProject.from_filename(filename)
        return cls(
            name=filename.name,  # TODO
            solution_guid="",  # TODO
            saved_path=filename,  # TODO
            local_path=filename,
            guid=loaded.guid,
            loaded=loaded,
        )

    def load(self) -> TwincatTsProject:
        if self.loaded is not None:
            return self.loaded

        if self.local_path is None:
            raise FileNotFoundError(
                f"File from project settings not found: {self.saved_path}"
            )

        if self.local_path.suffix.lower() in (".tsproj",):
            self.loaded = TwincatTsProject.from_filename(self.local_path)
            return self.loaded

        raise NotImplementedError(f"Format not yet supported: {self.local_path.suffix}")


@dataclasses.dataclass
class Solution:
    file_extension: ClassVar[str] = ".sln"
    root: pathlib.Path
    projects: list[Project]
    filename: Optional[pathlib.Path] = None

    @property
    def projects_by_name(self) -> dict[str, Project]:
        return {project.name: project for project in self.projects}

    @classmethod
    def from_projects(
        cls: type[Self],
        root: pathlib.Path,
        projects: List[pathlib.Path],
    ) -> Self:
        return cls(
            root=root,
            filename=None,
            projects=[Project.from_filename(proj) for proj in projects],
        )

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


def make_solution_from_files(filename: AnyPath) -> Solution:
    """
    From a TwinCAT solution (.sln) or .tsproj, get a Solution instance.

    Returns
    -------
    Solution
    """
    abs_path = pathlib.Path(filename).resolve()
    if abs_path.suffix.lower() == ".sln":
        return Solution.from_filename(abs_path)

    return Solution.from_projects(root=abs_path.parent, projects=[abs_path])


def single_file_loader(
    filename: pathlib.Path,
) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    source = TcSource.from_filename(filename)
    if source is None:
        logger.warning("No source found in file %s (is this in error?)", filename)
        return []
    return source.to_blark()


def get_blark_input_from_solution(
    solution: Solution,
) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    all_source = []
    for project in solution.projects:
        project = project.load()
        for plc in project.plcs:
            for source in plc.sources:
                all_source.append(source)

    inputs = []
    for item in all_source:
        if item.contents is not None:
            inputs.extend(item.contents.to_blark())
    return inputs


def project_loader(
    filename: pathlib.Path,
) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    solution = Solution.from_projects(filename.parent, [filename])
    return get_blark_input_from_solution(solution)


def solution_loader(
    filename: pathlib.Path,
) -> List[Union[BlarkSourceItem, BlarkCompositeSourceItem]]:
    solution = Solution.from_filename(filename)
    return get_blark_input_from_solution(solution)


register_file_handler(TcPOU.file_extension, single_file_loader)
register_file_handler(TcGVL.file_extension, single_file_loader)
register_file_handler(TcIO.file_extension, single_file_loader)
register_file_handler(TcDUT.file_extension, single_file_loader)
register_file_handler(Solution.file_extension, solution_loader)
register_file_handler(TwincatTsProject.file_extension, project_loader)
register_file_handler(TwincatPlcProject.file_extension, project_loader)
