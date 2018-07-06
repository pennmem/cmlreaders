def _import_readers():
    import importlib
    import pkgutil

    loader = pkgutil.get_loader("cmlreaders")
    module_info = [info for info in pkgutil.walk_packages(loader.path)
                   if info[1].startswith("cmlreaders.readers.")]
    modules = [
        importlib.import_module("." + info.name.split(".")[-1],
                                package="cmlreaders.readers")
        for info in module_info
    ]

    for module in modules:
        things = [getattr(module, thing) for thing in dir(module)
                  if thing.endswith("Reader")]

        for thing in things:
            globals()[thing.__name__] = thing


_import_readers()
