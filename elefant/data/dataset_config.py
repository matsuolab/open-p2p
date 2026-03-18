"""Dataset config types with no elefant_rust/zmq dependencies.

Extracted so config.py and inference can load these without pulling in
video_proto_dataset (which depends on zmq_queue and elefant_rust).
"""

from typing import List, Optional

from elefant.config import ConfigBase


class RandAugmentationConfig(ConfigBase):
    # The fraction of examples that will be augmented.
    fraction_augmented: float = 0.0
    augmentations: List[str] = []


class VideoProtoDatasetConfig(ConfigBase):
    local_prefix: str

    frame_height: int = 192
    frame_width: int = 192
    # Number of frames to return in each chunk.
    T: int = 10
    always_labelled: bool = False

    n_preprocess_workers_per_iter_worker: int = 8
    preprocessed_chunks_queue_size: int = 1024

    shuffle_rng_seed: int = 43
    shuffle: bool = True
    shuffle_buffer_size: int = 1024

    warn_on_starvation: bool = True

    rand_augmentation: RandAugmentationConfig = RandAugmentationConfig()

    # This is a hacky workaround that pytorch lightning doesn't seem to allow validation iterator to keep going
    # if you limit the number of validation batches.
    # So we just ignore the iterator
    ignore_iterator_reset: bool = False

    batch_size: int = 32
    dataset_worker_prefetch_factor: int = 2
    dataset_worker_num_workers: int = 1
    shuffled_chunks_queue_size: int = 16

    # For IPC we need a unique id for the dataset.
    # when using multi-GPU the unique id for the same dataset must be the same across all workers.
    # This will get modified by appending a randomly generated string to the end.
    dataset_unique_id: str

    load_video_name: Optional[str] = None

    def get_load_video_name(self):
        if self.load_video_name is None:
            return f"{self.frame_height}x{self.frame_width}.mp4"
        else:
            return self.load_video_name
