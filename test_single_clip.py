import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from encoders.silhouette_encoder import SilhouetteEncoder
from gallery.gallery             import GaitGallery
from enroll_and_test             import extract_seg_masks

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    ap = argparse.ArgumentParser(description="Test a single video clip against the Gait Gallery")
    ap.add_argument("video_path", help="Path to the video file to test")
    ap.add_argument("--camera-id", default=None, help="Optional camera ID for viewpoint-specific recognition")
    ap.add_argument("--config",   default="configs/model_config.yaml")
    ap.add_argument("--max-frames", type=int, default=64)
    args = ap.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        sys.exit(f"Error: File not found -> {video_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    device    = cfg["model"].get("device", "cpu")
    threshold = cfg["identification"]["unknown_threshold"]

    print(f"Loading models to {device}...")
    seg_model = YOLO("yolov8n-seg.pt")
    seg_model.to(device)
    sil_enc   = SilhouetteEncoder(
                    cfg["model"].get("silhouette_weights", "weights/deepgaitv2.pt"),
                    device=device)
    gallery   = GaitGallery(
                    cfg["model"].get("gallery_index", "weights/gallery.faiss"),
                    cfg["model"].get("gallery_meta",  "weights/gallery_meta.json"),
                    camera_id=args.camera_id)

    gallery.load()

    if gallery.size() == 0:
        sys.exit("Error: Gallery is empty. Please run enroll_and_test.py first to enroll subjects.")

    print(f"\nProcessing video: {video_path.name}")
    sil_seq = extract_seg_masks(video_path, seg_model, args.max_frames)

    if sil_seq is None:
        sys.exit("\n❌ No person detected consistently enough in the video to analyze gait.")

    print("Encoding gait silhouette sequence...")
    emb = sil_enc.encode(sil_seq)
    
    # Search gallery
    matches = gallery.search(emb, top_k=1)
    
    print("\n" + "="*50)
    print("IDENTIFICATION RESULT")
    print("="*50)
    
    if not matches:
        print("Result: UNKNOWN (No match found in gallery)")
    else:
        match = matches[0]
        if match.score >= threshold:
            print(f"Result: {match.person_id} ({match.name})")
            print(f"Confidence (Cosine Similarity): {match.score:.4f}")
            print(f"Threshold: > {threshold:.2f}")
            print("Status: ✅ KNOWN MATCH")
        else:
            print(f"Result: UNKNOWN")
            print(f"Best Match was: {match.person_id} / {match.name}")
            print(f"Confidence (Cosine Similarity): {match.score:.4f}")
            print(f"Threshold: < {threshold:.2f}")
            print("Status: ❌ BELOW THRESHOLD")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
