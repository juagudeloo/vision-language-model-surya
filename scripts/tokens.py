"""tokens.py — Crop Surya's token grid to a region and pool it down to a fixed number of tokens.

Moved unchanged from notebook 02 (cells 10 and 19) so scripts can import them; the notebook explains
why the crop happens on tokens and not on the image.
"""

import torch.nn.functional as F


def crop_tokens(tokens, bbox_px, patch_size=16, grid_size=256):
    """bbox_px: (x_min, y_min, x_max, y_max) en pixeles sobre la imagen 4096x4096."""
    x_min, y_min, x_max, y_max = bbox_px
    tx_min, ty_min = x_min // patch_size, y_min // patch_size
    tx_max, ty_max = -(-x_max // patch_size), -(-y_max // patch_size)  # techo (ceil)
    grid = tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[-1])
    cropped = grid[:, ty_min:ty_max, tx_min:tx_max, :]
    return cropped  # (B, h_tokens, w_tokens, D)


def pool_tokens(cropped, out_size=16):
    """
    Pooling 2D on the tokens. The input is expected to be of shape (B, h, w, D) and the output will be of shape (B, out_size*out_size, D).
    Args:
        cropped (torch.Tensor): The cropped tokens of shape (B, h, w, D).
        out_size (int): The desired output size for the pooled tokens. Default is 16.
    Returns:
        torch.Tensor: The pooled tokens of shape (B, out_size*out_size, D).
    """
    B, h, w, D = cropped.shape
    x = cropped.permute(0, 3, 1, 2)  # (B, D, h, w) -- pooling 2D espera canales primero
    x = F.adaptive_avg_pool2d(x, output_size=(out_size, out_size))
    x = x.permute(0, 2, 3, 1).reshape(B, out_size * out_size, D)
    return x  # (B, out_size*out_size, D)
