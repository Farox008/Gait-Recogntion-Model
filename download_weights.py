"""
download_weights.py

Downloads the official OpenGait pretrained checkpoints from HuggingFace
and saves them to the weights/ directory.

Then inspects their state dict keys so we can adapt our encoders.

Usage:
    python download_weights.py
"""

import os
import sys
import urllib.request
import ssl
import time

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

FILES = {
    "skeletongait++.pt": (
        "https://huggingface.co/opengait/OpenGait/resolve/main/"
        "Gait3D/SkeletonGaitPP/SkeletonGaitPP/checkpoints/SkeletonGaitPP-60000.pt"
    ),
    "deepgaitv2.pt": (
        "https://huggingface.co/opengait/OpenGait/resolve/main/"
        "Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt"
    ),
}


def _progress_hook(filename):
    last_time = [time.time()]
    def hook(count, block_size, total_size):
        now = time.time()
        if now - last_time[0] < 2.0 and count * block_size < total_size:
            return
        last_time[0] = now
        downloaded = count * block_size
        pct = min(100.0, 100.0 * downloaded / total_size) if total_size > 0 else 0
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        print(f"\r  {filename}: {mb_done:.1f} / {mb_total:.1f} MB  ({pct:.1f}%)", end="", flush=True)
    return hook


def download():
    # Allow unverified SSL for HuggingFace (some corporate proxies break it)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for name, url in FILES.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  [SKIP] {name} already exists ({size_mb:.1f} MB)")
            continue

        print(f"\nDownloading {name} ...")
        tmp = dest + ".tmp"
        try:
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook(name))
            print()
            os.rename(tmp, dest)
            print(f"  [OK] Saved to {dest}")
        except Exception as exc:
            print(f"\n  [ERROR] Failed to download {name}: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
            sys.exit(1)


def inspect():
    import torch
    print("\n" + "=" * 60)
    print("CHECKPOINT KEY INSPECTION")
    print("=" * 60)

    for name in FILES:
        path = os.path.join(WEIGHTS_DIR, name)
        if not os.path.isfile(path):
            print(f"\n  {name}: NOT FOUND — download first.")
            continue

        print(f"\n── {name} ──")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"   Top-level type: {type(ckpt)}")

        if isinstance(ckpt, dict):
            print(f"   Top-level keys: {list(ckpt.keys())[:20]}")
            # Drill into common wrapper keys
            for wrapper_key in ["model", "state_dict", "model_state_dict", "network"]:
                if wrapper_key in ckpt:
                    sd = ckpt[wrapper_key]
                    keys = list(sd.keys())
                    print(f"   [{wrapper_key}] — {len(keys)} keys")
                    print(f"   First 20 keys:")
                    for k in keys[:20]:
                        shape = tuple(sd[k].shape) if hasattr(sd[k], 'shape') else '?'
                        print(f"     {k:<60s}  {shape}")
                    break
            else:
                # No standard wrapper — it IS the state dict
                keys = list(ckpt.keys())
                print(f"   Direct state dict — {len(keys)} keys")
                print(f"   First 20 keys:")
                for k in keys[:20]:
                    shape = tuple(ckpt[k].shape) if hasattr(ckpt[k], 'shape') else '?'
                    print(f"     {k:<60s}  {shape}")
        else:
            print(f"   (non-dict checkpoint, type={type(ckpt)})")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenGait Weight Downloader")
    print("=" * 60)
    download()
    print("\nAll downloads complete. Inspecting checkpoints ...")
    inspect()
    print("\nDone. Share the key inspection output above so we can adapt the encoders.")
