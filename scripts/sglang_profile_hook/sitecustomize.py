"""Opt-in, process-local instrumentation; never edits the installed runtime."""
import importlib.abc
import importlib.machinery
import os
import sys


if os.environ.get("DFLASH_COMPONENT_PROFILE") in {"events", "trace"}:
    class DFlashProfileFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.speculative.dflash_worker":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Pinned DFlash worker not found")
            original = spec.loader.exec_module

            def instrument(module):
                original(module)
                from runtime_hooks import install
                install(module)

            spec.loader.exec_module = instrument
            return spec

    sys.meta_path.insert(0, DFlashProfileFinder())
