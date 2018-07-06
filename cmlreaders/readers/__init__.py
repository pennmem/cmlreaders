import importlib
import pkgutil
import sys

_mod = sys.modules[__name__]

loader = pkgutil.get_loader("cmlreaders")
module_info = [info for info in pkgutil.walk_packages(loader.path)
               if info.name.startswith("cmlreaders.readers.")]
modules = [
    importlib.import_module("." + info.name.split(".")[-1],
                            package="cmlreaders.readers")
    for info in module_info
]

for module in modules:
    things = [getattr(module, thing) for thing in dir(module)
              if thing.endswith("Reader")]

    for thing in things:
        setattr(_mod, thing.__name__, thing)
