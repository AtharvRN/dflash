"""Opt-in standalone benchmark after the recovered DFlash worker is loaded."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.environ.get('DFLASH_MIDVERIFY_LATENCY_CONFIG'):
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'sglang.srt.speculative.dflash_worker_v2':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original = spec.loader.exec_module

            def execute(module):
                original(module)
                init = module.DFlashWorkerV2.__init__

                def initialize(self, *args, **kwargs):
                    init(self, *args, **kwargs)
                    from latency_runtime import run
                    run(self, os.environ['DFLASH_MIDVERIFY_LATENCY_CONFIG'])

                module.DFlashWorkerV2.__init__ = initialize

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Finder())
