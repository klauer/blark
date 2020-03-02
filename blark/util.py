import pathlib

import pytmc


def get_source_code(fn):
    fn = pathlib.Path(fn)
    root = pytmc.parser.parse(str(fn))
    for item in root.find(pytmc.parser.TwincatItem):
        if hasattr(item, 'get_source_code'):
            return item.get_source_code()

    raise ValueError('Unable to find pytmc TwincatItem with source code '
                     '(i.e., with `get_source_code` as an attribute)')
