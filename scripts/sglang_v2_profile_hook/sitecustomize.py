"""Opt-in v2 instrumentation, loaded only in the separate event-timing run."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.environ.get("DFLASH_V2_PROFILE") == "1":
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.speculative.dflash_worker_v2":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Recovered DFlash v2 not found")
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                from v2_runtime_hooks import install
                install(module)

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Finder())
