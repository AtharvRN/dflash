"""Opt-in audit of the recovered worker; no installed files are modified."""
import importlib.abc
import importlib.machinery
import os
import sys


if os.environ.get("DFLASH_RAGGED_AUDIT_DIR"):
    class AuditFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sglang.srt.speculative.dflash_worker_v2":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                from audit_runtime import install
                install(module)

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, AuditFinder())
