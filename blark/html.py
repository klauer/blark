"""Syntax-highlighted HTML file writer."""
from __future__ import annotations

import collections
import dataclasses
import pathlib
from typing import Any, DefaultDict, Dict, List, Optional

import lark

from .output import OutputBlock, register_output_handler


@dataclasses.dataclass(frozen=True)
class HighlighterAnnotation:
    name: str
    terminal: bool
    is_open_tag: bool
    other_tag_pos: int

    def __str__(self) -> str:
        return self.as_string()

    def as_string(self, tag: str = "span") -> str:
        # some options here?
        if not self.is_open_tag:
            return f'</{tag}>'

        if self.terminal:
            classes = " ".join(("term", self.name))
        else:
            classes = " ".join(("rule", self.name))

        return f'<{tag} class="{classes}">'


def add_annotation_pair(
    annotations: DefaultDict[int, List[HighlighterAnnotation]],
    name: str,
    start_pos: int,
    end_pos: int,
    terminal: bool,
) -> None:
    annotations[start_pos].append(
        HighlighterAnnotation(
            name=name,
            terminal=terminal,
            is_open_tag=True,
            other_tag_pos=end_pos,
        )
    )
    annotations[end_pos].append(
        HighlighterAnnotation(
            name=name,
            terminal=terminal,
            is_open_tag=False,
            other_tag_pos=start_pos,
        )
    )


def get_annotations(tree: lark.Tree) -> DefaultDict[int, List[HighlighterAnnotation]]:
    """Get annotations for syntax elements in the given parse tree."""
    annotations: DefaultDict[int, List[HighlighterAnnotation]] = collections.defaultdict(
        list
    )

    for subtree in tree.iter_subtrees():
        add_annotation_pair(
            annotations,
            name=subtree.data,
            terminal=False,
            start_pos=subtree.meta.start_pos,
            end_pos=subtree.meta.end_pos,
        )
        for child in subtree.children:
            if isinstance(child, lark.Token):
                if child.start_pos is not None and child.end_pos is not None:
                    add_annotation_pair(
                        annotations,
                        name=child.type,
                        terminal=True,
                        start_pos=child.start_pos,
                        end_pos=child.end_pos,
                    )

        return annotations


def apply_annotations_to_code(
    code: str,
    annotations: Dict[int, List[HighlighterAnnotation]]
) -> str:
    result = []
    pos = 0
    for pos, ch in enumerate(code):
        for ann in annotations.get(pos, []):
            result.append(str(ann))
        result.append(ch)

    for ann in annotations.get(pos + 1, []):
        result.append(str(ann))

    html = "".join(result)
    html = f'<div class="blark-code">{html}</div>'
    html = html.replace("\n", "<br/>\n")
    # TODO remove
    html = """
    <style>
        .blark-code {
            font-family: SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",Courier,monospace;
        }
        .blark-code .term {
            font-weight: bold;
            color: #208050;
        }
        .INTEGER .SIGNED_INTEGER {
            color: red;
        }
        .set_statement .assignment_statement {
            font-weight: bold;
        }
    </style>
    """ + html  # noqa: E501
    return html


@dataclasses.dataclass
class HtmlWriter:
    user: Any
    source_filename: Optional[pathlib.Path]
    block: OutputBlock

    @property
    def source_code(self) -> str:
        assert self.block.origin is not None
        return self.block.origin.source_code

    def to_html(self) -> str:
        assert self.block.origin is not None
        assert self.block.origin.tree is not None
        annotations = get_annotations(self.block.origin.tree)

        for comment in self.block.origin.comments:
            # print(comment.start_pos, comment.line)
            # TODO
            if comment.start_pos is not None and comment.end_pos is not None:
                add_annotation_pair(
                    annotations,
                    name=comment.type,
                    start_pos=comment.start_pos,
                    end_pos=comment.end_pos,
                    terminal=True,
                )

        return apply_annotations_to_code(self.source_code, annotations)

    @staticmethod
    def save(
        user: Any,
        source_filename: Optional[pathlib.Path],
        parts: List[OutputBlock],
    ) -> str:
        result = []
        for part in parts:
            writer = HtmlWriter(user, source_filename, part)
            result.append(writer.to_html())

        return "\n\n".join(result)


def _register():
    """Register the HTML output file handlers."""
    register_output_handler("html", HtmlWriter.save)
    register_output_handler(".html", HtmlWriter.save)
