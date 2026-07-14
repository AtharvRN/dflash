__all__ = [
    "DFlashDraftModel",
    "DFlashSurvivalBlockPolicy",
    "DFlashV2HorizonBlockPolicy",
    "dflash_generate_dynamic",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    if name == "DFlashSurvivalBlockPolicy":
        from .policy import DFlashSurvivalBlockPolicy

        return DFlashSurvivalBlockPolicy

    if name == "DFlashV2HorizonBlockPolicy":
        from .policy import DFlashV2HorizonBlockPolicy

        return DFlashV2HorizonBlockPolicy

    if name == "dflash_generate_dynamic":
        from .dynamic import dflash_generate_dynamic

        return dflash_generate_dynamic

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
