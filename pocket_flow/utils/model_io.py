import torch


def _torch_load(path: str, map_location, *, weights_only: bool | None):
    """
    Call torch.load with an explicit weights_only flag when supported.
    Falls back to older torch versions that don't accept weights_only.
    """
    if weights_only is None:
        return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older torch: no weights_only kwarg
        return torch.load(path, map_location=map_location)


def load_ckpt(
    ckpt_path: str,
    device="cpu",
    *,
    prefer_weights_only: bool = True,
    allow_unsafe_fallback: bool = True,
):
    """
    Load a PocketFlow checkpoint.

    - First tries torch.load(..., weights_only=True) (when available) for safer loading.
    - If the checkpoint contains EasyDict, we allowlist it using torch.serialization's APIs.
    - If weights_only loading fails and allow_unsafe_fallback=True, fall back to
      weights_only=False (pickle-enabled, less safe) to preserve legacy behaviour.
    """
    map_location = device

    # Optional allowlisting for the weights_only loader.
    safe_globals = []
    try:
        from easydict import EasyDict  # imported lazily so this module stays lightweight
        safe_globals.append(EasyDict)
    except Exception:
        # If easydict isn't installed, we simply won't allowlist it.
        safe_globals = []

    if prefer_weights_only:
        try:
            if safe_globals and hasattr(torch, "serialization"):
                ser = torch.serialization

                # Prefer the scoped context-manager if available (avoids global side effects).
                if hasattr(ser, "safe_globals"):
                    with ser.safe_globals(safe_globals):  # type: ignore[attr-defined]
                        return _torch_load(ckpt_path, map_location, weights_only=True)

                # Otherwise, fall back to global allowlisting.
                if hasattr(ser, "add_safe_globals"):
                    ser.add_safe_globals(safe_globals)
                    return _torch_load(ckpt_path, map_location, weights_only=True)

            # No allowlisting available/needed.
            return _torch_load(ckpt_path, map_location, weights_only=True)

        except Exception:
            if not allow_unsafe_fallback:
                raise
            # Legacy behaviour: allow full pickle/unpickling.
            return _torch_load(ckpt_path, map_location, weights_only=False)

    # Explicit legacy path (also future-proof if torch flips the default).
    return _torch_load(ckpt_path, map_location, weights_only=False)


def load_model_from_ckpt(model_cls, ckpt_path, device):
    """Load a model and weights from a checkpoint."""
    ckpt = load_ckpt(ckpt_path, device)
    config = ckpt["config"]
    model = model_cls(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model
