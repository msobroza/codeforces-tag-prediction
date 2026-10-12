"""The device a method that runs torch runs on, chosen where it is deployed: ``$CODEFORCES_DEVICE``.

A method that runs torch (a ModernBERT or an F2LLM method) asks ``requested_device`` for its
device setting when it loads, so that one method folder, and one image, runs on the CPU or on
the GPU as its deployment says. The variable is read at call time (as ``read_samples`` reads
``$CODEFORCES_DATA_DIR``): unset or empty, it leaves the caller's default. A value outside
``config.DEVICES`` is refused as written (no case folding, no stripping), and so is the variable
the F2LLM methods read before, ``config.LEGACY_DEVICE_ENV_VAR``, whenever it is set: a device
variable silently ignored, on a machine whose GPU once rebooted it, is the failure to avoid.

Turning a setting into a device (``"auto"`` into CUDA or the CPU) is each method's own
``resolve_device``. This module imports only the standard library and ``codeforces_tags.config``,
so the core imports it without torch.
"""

from __future__ import annotations

import os

from codeforces_tags.config import DEVICE_ENV_VAR, DEVICES, LEGACY_DEVICE_ENV_VAR


def requested_device(default: str = "auto") -> str:
    """Return the device setting that a method running torch runs on, read at call time.

    Args:
        default: The setting when ``$CODEFORCES_DEVICE`` is unset or empty: ``"auto"``, or a
            method's own setting (a ModernBERT method's ``device``).

    Returns:
        ``$CODEFORCES_DEVICE`` when it is set and not empty, else ``default``: one of
        ``config.DEVICES``, ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Raises:
        ValueError: If ``$CODEFORCES_F2LLM_DEVICE`` (``config.LEGACY_DEVICE_ENV_VAR``) is set and
            not empty, whatever ``$CODEFORCES_DEVICE`` holds (the message names the new
            variable), or if ``default`` or ``$CODEFORCES_DEVICE`` is not exactly one of
            ``config.DEVICES``.
    """
    if os.environ.get(LEGACY_DEVICE_ENV_VAR):
        raise ValueError(
            f"${LEGACY_DEVICE_ENV_VAR} is no longer read: unset it, and name the device in "
            f"${DEVICE_ENV_VAR} instead, one of {DEVICES}."
        )
    _check_setting("The default device", default)
    requested = os.environ.get(DEVICE_ENV_VAR, "")
    if not requested:
        return default
    _check_setting(f"${DEVICE_ENV_VAR}", requested)
    return requested


def _check_setting(what: str, setting: str) -> None:
    """Refuse a device setting that is not exactly one of ``config.DEVICES``, naming its source."""
    if setting not in DEVICES:
        raise ValueError(f"{what} is {setting!r}, not a device setting: expected one of {DEVICES}.")
