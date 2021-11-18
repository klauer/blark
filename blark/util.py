import pathlib

import pytmc


def get_source_code(fn):
    fn = pathlib.Path(fn)
    root = pytmc.parser.parse(str(fn))
    for item in root.find(pytmc.parser.TwincatItem):
        if hasattr(item, "get_source_code"):
            return item.get_source_code()

    raise ValueError(
        "Unable to find pytmc TwincatItem with source code "
        "(i.e., with `get_source_code` as an attribute)"
    )


def indent_inner(text: str, prefix: str) -> str:
    """Indent the inner lines of ``text`` (not first and last) with ``prefix``."""
    lines = text.splitlines()
    if len(lines) < 3:
        return text

    return "\n".join(
        (
            lines[0],
            *(f"{prefix}{line}" for line in lines[1:-1]),
            lines[-1],
        )
    )


def python_debug_session(namespace, message):
    """
    Enter an interactive debug session with pdb or IPython, if available.
    """
    import blark  # noqa

    debug_namespace = dict(pytmc=pytmc, blark=blark)
    debug_namespace.update(
        **{k: v for k, v in namespace.items()
           if not k.startswith('__')}
    )
    globals().update(debug_namespace)

    print(
        "\n".join(
            (
                "-- blark debug --",
                message,
                "-- blark debug --",
            )
        )
    )

    try:
        from IPython import embed  # noqa
    except ImportError:
        import pdb  # noqa
        pdb.set_trace()
    else:
        embed()
