import pytmc


def get_source_code(fn):
    fn = str(fn)
    if fn.endswith('.TcPOU'):
        root = pytmc.parser.parse(fn)
        pou, = list(root.find(pytmc.parser.POU))
        source_code = pou.get_source_code()
    else:
        with open(fn) as f:
            source_code = f.read()

    return source_code
