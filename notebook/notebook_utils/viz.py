"""viz.py — Small helpers for pulling displayable slices out of Surya's tensors."""

import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as sunpy_cm
from matplotlib.colors import Colormap
from torch import Tensor

# Real SDO colormaps per channel, verified in section 7's 4x4 channel grid
# (docs/surya_forward_pass_study_plan.md) and in downstream_examples/ar_segmentation/infer.py
# of the Surya repo itself -- not invented here. HMI channels other than hmi_v (a signed
# velocity, not a field) all share "hmimag".
CHANNEL_CMAPS: dict[str, Colormap] = {
    "aia94": sunpy_cm.cmlist["sdoaia94"], "aia131": sunpy_cm.cmlist["sdoaia131"],
    "aia171": sunpy_cm.cmlist["sdoaia171"], "aia193": sunpy_cm.cmlist["sdoaia193"],
    "aia211": sunpy_cm.cmlist["sdoaia211"], "aia304": sunpy_cm.cmlist["sdoaia304"],
    "aia335": sunpy_cm.cmlist["sdoaia335"], "aia1600": sunpy_cm.cmlist["sdoaia1600"],
    "hmi_m": sunpy_cm.cmlist["hmimag"], "hmi_bx": sunpy_cm.cmlist["hmimag"],
    "hmi_by": sunpy_cm.cmlist["hmimag"], "hmi_bz": sunpy_cm.cmlist["hmimag"],
    "hmi_v": plt.get_cmap("bwr"),
}

# Friendly aliases for the HMI channels, whose config["data"]["sdo_channels"] names
# ("hmi_m", "hmi_v", ...) are not self-explanatory the way "aia304" is.
_HMI_CHANNEL_ALIASES = {
    "magnetogram": "hmi_m",
    "dopplergram": "hmi_v",
    "bx": "hmi_bx",
    "by": "hmi_by",
    "bz": "hmi_bz",
}


def resolve_channel_name(channel: str, config: dict) -> str:
    """Resolve a short/friendly channel name to its real config["data"]["sdo_channels"] entry.

    Args:
        channel: Either the exact name already in config["data"]["sdo_channels"]
            (e.g. "aia304", "hmi_m"), a bare AIA wavelength ("304", "1600"),
            or a friendly HMI alias ("magnetogram", "dopplergram", "bx"/"by"/"bz").
        config: Parsed config.yaml, for its "data"."sdo_channels" list.

    Returns:
        The exact channel name as it appears in config["data"]["sdo_channels"].

    Raises:
        ValueError: If channel can't be resolved to any known entry.

    Example:
        >>> resolve_channel_name("304", config)
        'aia304'
        >>> resolve_channel_name("magnetogram", config)
        'hmi_m'
    """
    channels = config["data"]["sdo_channels"]
    if channel in channels:
        return channel
    if channel.isdigit() and f"aia{channel}" in channels:
        return f"aia{channel}"
    alias = _HMI_CHANNEL_ALIASES.get(channel.lower())
    if alias and alias in channels:
        return alias
    raise ValueError(f"Could not resolve channel {channel!r} against {channels}.")


def get_channel_image(ts: Tensor, config: dict, channel: str, timestep: int = -1) -> Tensor:
    """Extract one channel's 2D image from a single sample's "ts" tensor, for display.

    Takes ts already indexed down to one sample -- e.g. batch_data["ts"][0]
    for SuryaBatch.load()'s single-sample batches, or batch_data["ts"][k] for
    sample k of a batch with batch_size > 1 -- rather than hardcoding which
    sample to use internally. Which sample to look at is the caller's choice,
    made once, before calling this.

    Args:
        ts: A single sample's tensor, shape (C, T, H, W) -- not the raw batch
            dict, and not batch_data["ts"] with its batch axis still on it.
        config: Parsed config.yaml -- looks up the channel's index in
            config["data"]["sdo_channels"], the same order ts's channel axis
            follows.
        channel: A channel name or friendly alias -- see resolve_channel_name.
        timestep: Which of the T input frames to use. -1 (default) is the
            most recent one -- with this project's [-60, 0] minute setup,
            that's the "0 minutes before" reference frame (index 1), not the
            one an hour earlier (index 0, "-60"). See section 5's
            time_delta_input explanation.

    Returns:
        A 2D tensor, shape (H, W).

    Example:
        >>> from notebook_utils.dataset import SuryaBatch
        >>> from notebook_utils.viz import get_channel_image
        >>> batch = SuryaBatch.load()
        >>> ts = batch.batch_data["ts"][0]  # sample 0 of this (size-1) batch
        >>> get_channel_image(ts, batch.config, "304").shape
        torch.Size([4096, 4096])
    """
    channel = resolve_channel_name(channel, config)
    channel_idx = config["data"]["sdo_channels"].index(channel)
    return ts[channel_idx, timestep]


def channel_display_range(arr: np.ndarray, channel: str) -> tuple[float, float]:
    """Real vmin/vmax for a channel's 2D array, robust to the off-disk fill value.

    Same logic validated in section 7's real 4x4 channel grid: HMI channels
    (signed field/velocity) are centered symmetrically at 0 using the 98th
    percentile of |on-disk value|; other channels use a plain 2nd-98th
    percentile stretch. Both exclude the off-disk fill value (read from the
    array's own corner, since it's a constant) from the statistics -- left
    in, it dominates the percentiles and washes out the real dynamic range.

    Args:
        arr: A 2D numpy array, one channel's image (already .cpu().numpy()'d).
        channel: The real config["data"]["sdo_channels"] name -- resolve
            first with resolve_channel_name if you only have a short alias.

    Returns:
        (vmin, vmax), ready to pass to imshow.
    """
    fill_value = arr[0, 0]
    on_disk = arr[arr != fill_value]
    if channel.startswith("hmi"):
        vabs = np.percentile(np.abs(on_disk), 98)
        return -vabs, vabs
    vmin, vmax = np.percentile(on_disk, [2, 98])
    return vmin, vmax


def get_channel_display(
    ts: Tensor, config: dict, channel: str, timestep: int = -1
) -> tuple[Tensor, Colormap, float, float]:
    """Everything imshow needs for one channel: the image, its real colormap, and vmin/vmax.

    Combines get_channel_image, CHANNEL_CMAPS, and channel_display_range so
    a caller only has to pick the channel -- cmap and intensity scaling are
    already settled, the same way section 7's 4x4 grid did it.

    Args:
        ts: A single sample's tensor, shape (C, T, H, W).
        config: Parsed config.yaml.
        channel: A channel name or friendly alias -- see resolve_channel_name.
        timestep: Which of the T input frames to use, see get_channel_image.

    Returns:
        (image, cmap, vmin, vmax) -- pass straight to
        imshow(image.cpu(), cmap=cmap, vmin=vmin, vmax=vmax).

    Example:
        >>> image, cmap, vmin, vmax = get_channel_display(ts, config, "304")
        >>> ax.imshow(image.cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    """
    resolved = resolve_channel_name(channel, config)
    image = get_channel_image(ts, config, resolved, timestep)
    vmin, vmax = channel_display_range(image.cpu().numpy(), resolved)
    return image, CHANNEL_CMAPS[resolved], vmin, vmax
