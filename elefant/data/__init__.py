from importlib import import_module

from elefant.data.action_mapping import (
    StructuredAction,
    UniversalAutoregressiveActionMapping,
    UniversalAutoregressiveActionMappingConfig,
)

__all__ = [
    "ActionLabelVideoDatasetItem",
    "ActionLabelVideoProtoDataset",
    "ActionLabelVideoProtoDatasetConfig",
    "RandAugmentationConfig",
    "StructuredAction",
    "VideoProtoDatasetConfig",
    "UniversalAutoregressiveActionMappingConfig",
    "UniversalAutoregressiveActionMapping",
    "DummyDataset",
    "DummyDatasetConfig",
]


_LAZY_IMPORTS = {
    "ActionLabelVideoDatasetItem": (
        "elefant.data.action_label_video_proto_dataset",
        "ActionLabelVideoDatasetItem",
    ),
    "ActionLabelVideoProtoDataset": (
        "elefant.data.action_label_video_proto_dataset",
        "ActionLabelVideoProtoDataset",
    ),
    "ActionLabelVideoProtoDatasetConfig": (
        "elefant.data.action_label_video_proto_dataset",
        "ActionLabelVideoProtoDatasetConfig",
    ),
    "RandAugmentationConfig": (
        "elefant.data.dataset_config",
        "RandAugmentationConfig",
    ),
    "VideoProtoDatasetConfig": (
        "elefant.data.dataset_config",
        "VideoProtoDatasetConfig",
    ),
    "DummyDataset": ("elefant.data.dummy_dataset", "DummyDataset"),
    "DummyDatasetConfig": ("elefant.data.dummy_dataset", "DummyDatasetConfig"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
