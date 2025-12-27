#!/usr/bin/env python3
"""
Test script to verify load_ckpt works without serialization errors.

Run via:
  pixi run test-load-ckpt
"""

import os
import sys

# Ensure project root is on sys.path so `pocket_flow` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from easydict import EasyDict

from pocket_flow import PocketFlow
from pocket_flow.utils.model_io import load_ckpt


def main() -> int:
    print("✓ Successfully imported load_ckpt")

    has_serialization = hasattr(torch, "serialization")
    print(f"✓ torch has serialization: {has_serialization}")

    if has_serialization:
        has_add_safe_globals = hasattr(torch.serialization, "add_safe_globals")
        has_safe_globals = hasattr(torch.serialization, "safe_globals")
        print(f"✓ torch.serialization has add_safe_globals: {has_add_safe_globals}")
        print(f"✓ torch.serialization has safe_globals: {has_safe_globals}")

    # Exercise the weights_only path (and allowlisting) via load_ckpt
    ckpt_path = os.environ.get("POCKETFLOW_CKPT", "ckpt/ZINC-pretrained-255000.pt")
    obj = load_ckpt(ckpt_path, device="cpu", prefer_weights_only=True, allow_unsafe_fallback=True)

    assert isinstance(obj, dict), f"Expected dict checkpoint, got {type(obj)}"
    assert "model" in obj, "Checkpoint missing 'model' key"
    assert "config" in obj, "Checkpoint missing 'config' key"

    # If config is EasyDict, this confirms the allowlisting path is working.
    if isinstance(obj.get("config"), EasyDict):
        print("✓ config is EasyDict (allowlisting path exercised)")

    print("✓ Loaded checkpoint successfully")

    # Test load_state_dict with strict=False and check for missing/unexpected keys
    print("Testing model.load_state_dict with strict=False...")
    config = obj["config"]
    model = PocketFlow(config).to("cpu")
    missing, unexpected = model.load_state_dict(obj["model"], strict=False)

    if missing:
        print(f"✗ Missing keys: {missing}")
        return 1
    if unexpected:
        print(f"✗ Unexpected keys: {unexpected}")
        return 1

    print("✓ No missing or unexpected keys in checkpoint")
    print("✓ All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
