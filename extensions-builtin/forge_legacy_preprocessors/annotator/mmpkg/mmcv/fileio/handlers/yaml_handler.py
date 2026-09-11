# Copyright (c) OpenMMLab. All rights reserved.
import yaml

try:
    from yaml import CSafeDumper as Dumper
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeDumper as Dumper
    from yaml import SafeLoader as Loader

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        kwargs.pop("Loader", None)
        loader = Loader(file, **kwargs)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
