import ctypes
from functools import lru_cache
from typing import Tuple

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


@lru_cache(maxsize=1)
def _get_rust_module():
    try:
        import elefant_rust
    except ImportError:
        return None
    return elefant_rust


def _canonical_resize(im: torch.Tensor, exp_dim: Tuple[int, int]) -> torch.Tensor:
    """Wrap the rust code in something safer that deals with torch tensors.

    Only works with 3-channel RGB images.
    """
    # Torch tensors are CHW
    assert im.shape[0] == 3

    # Convert the tensor to HWC and then a bytes object.
    im_hwc = im.permute(1, 2, 0)
    im_hwc = im_hwc.contiguous().cpu()
    ptr = im_hwc.data_ptr()
    num_bytes = im_hwc.numel() * im_hwc.element_size()
    im_bytes_in = ctypes.string_at(ptr, num_bytes)

    elefant_rust = _get_rust_module()
    if elefant_rust is None:
        resized_im = F.resize(
            im,
            list(exp_dim),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if resized_im.dtype != torch.uint8:
            resized_im = resized_im.round().clamp(0, 255).to(torch.uint8)
        return resized_im

    resized_im = elefant_rust.resize_image(im_bytes_in, *im.shape[1:], *exp_dim)

    # Convert the bytes object back to a tensor.
    im_out = torch.frombuffer(resized_im, dtype=torch.uint8)
    im_out = im_out.view(exp_dim[0], exp_dim[1], 3)
    im_out = im_out.permute(2, 0, 1)
    return im_out


def resize_image_for_model(im: torch.Tensor, inp_dim) -> torch.Tensor:
    assert len(im.shape) == 3
    assert im.shape[0] == 3

    if im.shape[1] == inp_dim[0] and im.shape[2] == inp_dim[1]:
        return im
    resized_im = _canonical_resize(im, inp_dim)
    return resized_im
