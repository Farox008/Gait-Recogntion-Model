"""
enroll_and_test.py  (v4 — YOLOv8-Seg silhouette + DeepGaitV2)

Key changes vs v3:
  • YOLOv8n-seg produces proper binary person segmentation masks per frame
  • Masks are normalised/resized to (128, 88) matching DeepGaitV2 training format
  • Morphological cleanup applied to remove noise
  • Enrollment uses ALL clips per angle (average over many embeddings)

Usage:
  python enroll_and_test.py --vods-dir ../vods
"""

import argparse, logging, sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from encoders.silhouette_encoder import SilhouetteEncoder
from gallery.gallery             import GaitGallery

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enroll_test")

MASK_H, MASK_W = 128, 88
PERSON_DISPLAY  = {"person_1": "Person A", "person_2": "Person B"}

# ── Segmentation-based silhouette extraction ─────────────────────────────────

def extract_seg_masks(video_path: Path, seg_model: YOLO,
                      max_frames: int = 64) -> np.ndarray | None:
    """
    Run YOLOv8-seg on evenly-sampled frames, extract the largest person mask
    per frame, resize to (MASK_H, MASK_W), return (T, H, W) uint8 array.
    Returns None if no person detected in enough frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = set(np.linspace(0, total - 1, min(max_frames, total), dtype=int))
    frames  = []
    idx     = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        return None

    masks = []
    blank = np.zeros((MASK_H, MASK_W), dtype=np.uint8)

    for frame in frames:
        results = seg_model.predict(frame, classes=[0], conf=0.35, verbose=False)
        r = results[0]
        if r.masks is None or len(r.masks) == 0:
            masks.append(blank)
            continue

        # Pick the largest mask (most pixels)
        person_masks = r.masks.data.cpu().numpy()  # (N, H', W')
        areas = [m.sum() for m in person_masks]
        best  = person_masks[int(np.argmax(areas))]  # (H', W') float [0,1]

        # Resize to target
        m8 = (best * 255).astype(np.uint8)
        m8 = cv2.resize(m8, (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel)
        _, m8 = cv2.threshold(m8, 127, 255, cv2.THRESH_BINARY)
        masks.append(m8)

    # Need at least 5 non-empty frames
    non_empty = sum(1 for m in masks if m.sum() > 0)
    if non_empty < 5:
        return None

    # Pad to max_frames if needed
    while len(masks) < max_frames:
        masks.append(blank)

    return np.stack(masks[:max_frames], axis=0)  # (T, H, W)


# ── Collect clips ─────────────────────────────────────────────────────────────

def collect_clips(vods_dir: Path):
    """
    Returns enroll and test dicts: {pid: [clip_paths]}
    Enrollment: Up to 3 clips per angle
    Test: ALL remaining clips
    """
    enroll, test = {}, {}
    for p in sorted(vods_dir.iterdir()):
        if not p.is_dir():
            continue
        pid = p.name
        enroll[pid] = []
        test[pid]   = []
        for a in sorted(p.iterdir()):
            if not a.is_dir():
                continue
            clips = sorted(a.glob("*.mp4"))
            if not clips:
                continue
            # Use up to 3 clips for a richer gallery embedding
            n_enroll = min(3, max(1, len(clips) // 2)) if len(clips) > 1 else 1
            if len(clips) >= 3:
                n_enroll = 3
                
            enroll[pid].extend(clips[:n_enroll])
            test[pid].extend(clips[n_enroll:])
    return enroll, test


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vods-dir",   default="../vods")
    ap.add_argument("--config",     default="configs/model_config.yaml")
    ap.add_argument("--max-frames", type=int,   default=64)
    ap.add_argument("--threshold",  type=float, default=None)
    ap.add_argument("--min-fill",   type=float, default=0.5,
                    help="Min fraction of frames that must contain a person for an enrol clip to be used (default 0.5)")
    ap.add_argument("--mode",       choices=["train", "test", "both"], default="both",
                    help="Mode to run: train (enroll), test (identify), or both (default)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    threshold = args.threshold or cfg["identification"]["unknown_threshold"]
    device    = cfg["model"].get("device", "cpu")

    vods_dir = Path(args.vods_dir).resolve()
    if not vods_dir.exists():
        sys.exit(f"vods dir not found: {vods_dir}")

    # Models
    print("Loading models …")
    seg_model = YOLO("yolov8n-seg.pt")
    seg_model.to(device)
    sil_enc   = SilhouetteEncoder(
                    cfg["model"].get("silhouette_weights", "weights/deepgaitv2.pt"),
                    device=device)
    gallery   = GaitGallery(
                    cfg["model"].get("gallery_index", "weights/gallery.faiss"),
                    cfg["model"].get("gallery_meta",  "weights/gallery_meta.json"))

    enroll_clips, test_clips = collect_clips(vods_dir)
    all_pids = sorted(set(list(enroll_clips) + list(test_clips)))

    # ── PHASE 1: Enrollment ───────────────────────────────────────────────────
    if args.mode in ["train", "both"]:
        print("\n" + "=" * 65)
        print("PHASE 1 — ENROLLMENT  (YOLOv8-Seg masks + DeepGaitV2)")
        print("=" * 65)

        for pid in all_pids:
            clips = enroll_clips.get(pid, [])
            name  = PERSON_DISPLAY.get(pid, pid)
            print(f"\n► {name} ({pid}) — {len(clips)} enrol clip(s)")

            embeddings = []
            for clip in clips:
                rel = f"{clip.parent.name}/{clip.name}"
                print(f"   {rel}: ", end="", flush=True)
                sil_seq = extract_seg_masks(clip, seg_model, args.max_frames)
                if sil_seq is None:
                    print("SKIP (no detection)")
                    continue
                non_empty = sum(1 for m in sil_seq if m.sum() > 0)
                fill_ratio = non_empty / len(sil_seq)
                if fill_ratio < args.min_fill:
                    print(f"BAD QUAL (non-empty frames: {non_empty}/{len(sil_seq)} = {fill_ratio:.0%} < {args.min_fill:.0%}) — skipped")
                    continue
                emb = sil_enc.encode(sil_seq)
                embeddings.append(emb)
                print(f"OK  (non-empty frames: {non_empty}/{len(sil_seq)} = {fill_ratio:.0%})")

            if not embeddings:
                print(f"   ⚠ No valid embeddings for {pid}!")
                continue

            mean_emb = np.mean(embeddings, axis=0)
            n = np.linalg.norm(mean_emb)
            mean_emb = mean_emb / n if n > 1e-8 else mean_emb
            gallery.enroll(pid, name, mean_emb)
            print(f"   ✓ Enrolled from {len(embeddings)}/{len(clips)} clip(s).")

        gallery.save()
        print(f"\nGallery saved — {gallery.size()} person(s).")
    else:
        print("\n" + "=" * 65)
        print("PHASE 1 — ENROLLMENT (SKIPPED)")
        print("=" * 65)

    # ── PHASE 2: Identification ───────────────────────────────────────────────
    if args.mode in ["test", "both"]:
        print("\n" + "=" * 65)
        print(f"PHASE 2 — IDENTIFICATION  (threshold={threshold:.2f})")
        print("=" * 65)

        correct, total = 0, 0
        per_person = {pid: {"c": 0, "t": 0, "scores": []} for pid in all_pids}

        for true_pid in all_pids:
            clips = test_clips.get(true_pid, [])
            if not clips:
                continue
            name = PERSON_DISPLAY.get(true_pid, true_pid)
            print(f"\n► Testing {name} ({len(clips)} probes)")

            for clip in clips:
                sil_seq = extract_seg_masks(clip, seg_model, args.max_frames)
                rel     = f"{clip.parent.name}/{clip.name}"

                if sil_seq is None:
                    print(f"   ? {rel:<45s} NO DETECTION")
                    continue

                emb     = sil_enc.encode(sil_seq)
                matches = gallery.search(emb, top_k=1)

                if not matches or matches[0].score < threshold:
                    pred_pid = "UNKNOWN"
                    score    = matches[0].score if matches else 0.0
                else:
                    pred_pid = matches[0].person_id
                    score    = matches[0].score

                ok = (pred_pid == true_pid)
                correct += int(ok)
                total   += 1
                per_person[true_pid]["c"] += int(ok)
                per_person[true_pid]["t"] += 1
                per_person[true_pid]["scores"].append(score)

                mark = "✓" if ok else "✗"
                print(f"   {mark} {rel:<45s} → {pred_pid:<12s} ({score:.4f})")

        # ── Results ───────────────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("RESULTS")
        print("=" * 65)
        acc = 100.0 * correct / total if total else 0.0
        print(f"  Overall accuracy : {correct}/{total}  ({acc:.1f}%)")
        print(f"  Threshold used   : {threshold}")
        print()
        for pid in all_pids:
            d = per_person[pid]
            if d["t"] == 0:
                continue
            avg_s = float(np.mean(d["scores"])) if d["scores"] else 0.0
            print(f"  {PERSON_DISPLAY.get(pid,pid):<12s} ({pid}): "
                  f"{d['c']}/{d['t']} correct,  avg score={avg_s:.4f}")
    else:
        print("\n" + "=" * 65)
        print("PHASE 2 — IDENTIFICATION (SKIPPED)")
        print("=" * 65)
    print()


if __name__ == "__main__":
    main()
