"""
api_server.py — FastAPI wrapper for GaitPipeline
Provides the exact API contract expected by the StepSecure React frontend
(replacing the old crop-based model_server/main.py).
"""
import os
import asyncio
import uuid
import tempfile
import cv2
import numpy as np
import torch
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("[model] Shutting down thread executors to release port...")
    import asyncio
    import concurrent.futures
    loop = asyncio.get_event_loop()
    if isinstance(loop._default_executor, concurrent.futures.ThreadPoolExecutor):
        loop._default_executor.shutdown(wait=False, cancel_futures=True)

# Import our new GaitPipeline components
from pipeline.detector import PersonDetector
from pipeline.tracker import GaitTracker
from encoders.silhouette_encoder import SilhouetteEncoder
from gallery.gallery import GaitGallery
from identification.identifier import GaitIdentifier
from clipper import frame_to_b64_jpeg

app = FastAPI(title="Gait Model (Path A)", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Model State ────────────────────────────────────────────────────────

device = "cuda" # Defaulting to CUDA

detector = PersonDetector(model="yolov8n-seg.pt", conf=0.4, device=device, detect_every=1)
tracker = GaitTracker(max_age=30)
sil_encoder = SilhouetteEncoder(weights_path="weights/deepgaitv2.pt", device=device)
gallery = GaitGallery(index_path="weights/gallery.faiss", meta_path="weights/gallery_meta.json")

# Ensure gallery is loaded
if os.path.exists("weights/gallery_meta.json"):
    gallery.load()


# ── Helper Functions (reusing logic from enroll_and_test.py) ─────────────────

def save_tempfile(data: bytes, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, data)
    os.close(fd)
    return path

def extract_seg_masks(video_path: str, max_frames=64, min_fill_ratio=0.5) -> np.ndarray:
    """Extract standard 64x64 silhouette masks directly from YOLOv8-Seg"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Calculate skip to roughly get max_frames evenly spaced
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total_frames // max_frames)
    
    masks_seq = []
    frame_idx = 0
    non_empty = 0

    while len(masks_seq) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % skip != 0:
            frame_idx += 1
            continue
            
        frame_idx += 1
        
        # YOLOv8-Seg returns masks
        results = detector.model(frame, verbose=False, classes=[0], conf=0.4)
        r = results[0]
        
        if r.masks is None or len(r.masks) == 0:
            masks_seq.append(np.zeros((64, 64), dtype=np.uint8))
            continue
            
        # Find largest mask by area
        areas = [len(m.xy[0]) for m in r.masks]
        best_idx = np.argmax(areas)
        
        # Get binary mask, scale to original frame size
        mask = r.masks.data[best_idx].cpu().numpy()
        
        # The mask is usually downsampled (e.g. 160x160), resize it matching the bbox crop
        box = r.boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        
        # We need to reshape/resize the full frame mask to get the crop
        full_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_crop = full_mask[y1:y2, x1:x2]
        
        if mask_crop.size == 0:
            masks_seq.append(np.zeros((64, 64), dtype=np.uint8))
            continue
            
        # Resize to standard 64x64
        mask_standard = cv2.resize(mask_crop, (64, 64), interpolation=cv2.INTER_NEAREST)
        mask_standard = (mask_standard * 255).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_standard = cv2.morphologyEx(mask_standard, cv2.MORPH_OPEN, kernel)
        mask_standard = cv2.morphologyEx(mask_standard, cv2.MORPH_CLOSE, kernel)
        
        # Centre the silhouette (simple center of mass approach)
        M = cv2.moments(mask_standard)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            shift_x = 32 - cX
            M_trans = np.float32([[1, 0, shift_x], [0, 1, 0]])
            mask_standard = cv2.warpAffine(mask_standard, M_trans, (64, 64))
            
        masks_seq.append(mask_standard)
        non_empty += 1

    cap.release()
    
    if len(masks_seq) < 10:
        return None
        
    # Quality filter: discard clips where person is detected in < threshold of sampled frames
    fill_ratio = non_empty / len(masks_seq)
    if fill_ratio < min_fill_ratio:
        return None

    # Stack to shape (T, H, W) -> (T, 64, 64)
    return np.stack(masks_seq)

def process_registration(temp_paths: List[str], person_id: str, name: str):
    """Process N videos, embed using DeepGaitV2, and enrol."""
    all_embeddings = []
    frames_checked = 0
    valid_clips = 0

    for path in temp_paths:
        seq = extract_seg_masks(path, max_frames=64, min_fill_ratio=0.5)
        if seq is None:
            continue
            
        frames_checked += len(seq)
        
        emb_proj = sil_encoder.encode(seq)
        
        # Since fusion is NOT used in Path A tests, we just use the sil branch
        # zero-pad skeleton side (or just return sil) -> identifier expects 512d 
        # for backwards compatibility with GaitIdentifier or we can just enrol 256d
        # In our enroll_and_test.py, we directly save sil_emb
        padded_emb = np.zeros(512, dtype=np.float32)
        padded_emb[256:] = emb_proj
        
        # Normalize
        norm = np.linalg.norm(padded_emb)
        if norm > 0:
            padded_emb = padded_emb / norm
            
        all_embeddings.append(padded_emb)
        valid_clips += 1

    if not all_embeddings:
        return {
            "person_id": person_id,
            "name": name,
            "quality_score": 0,
            "frames_checked": frames_checked,
            "embeddings_extracted": 0,
            "enrolled": False,
            "message": "Quality too low (fill < 50%) or no person detected."
        }

    # Average
    avg_emb = np.mean(all_embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    gallery.enroll(person_id, name, avg_emb, confidence_score=0.9, video_count=valid_clips)
    gallery.save()

    return {
        "person_id": person_id,
        "name": name,
        "quality_score": min(95.0, 50.0 + (valid_clips * 10)), # heuristic
        "frames_checked": frames_checked,
        "embeddings_extracted": valid_clips,
        "enrolled": True,
        "message": f"Enrolled successfully using {valid_clips} clips."
    }

def process_test(video_path: str):
    """Run full pipeline on single video to identify tracks."""
    import time
    start = time.time()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    
    # Simple hack: just treat the whole video as one track for now if it's a test clip.
    # In a real test, we'd use track.update(). For `enroll_and_test.py`, test clips
    # are just fed to `extract_seg_masks` completely. Let's do that for simplicity + accuracy.
    # Wait, the frontend wants per-track. We will use the exact logic from `enroll_and_test.py`
    # and just output 1 "track" (the main person).
    
    seq = extract_seg_masks(video_path, max_frames=64, min_fill_ratio=0.1)
    
    # Also get a representative frame for thumbnail
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, frame = cap.read()
    thumb_b64 = frame_to_b64_jpeg(frame, quality=70) if ret else ""
    cap.release()

    persons_detected = []
    
    if seq is not None:
        total_frames = len(seq)
        
        emb_proj = sil_encoder.encode(seq)
        padded_emb = np.zeros(512, dtype=np.float32)
        padded_emb[256:] = emb_proj
        
        norm = np.linalg.norm(padded_emb)
        if norm > 0:
            padded_emb = padded_emb / norm
            
        matches = gallery.search(padded_emb, top_k=1)
        
        verdict = "UNKNOWN"
        confidence = 0
        pid = None
        pname = None
        score = 0
        
        if matches:
            best = matches[0]
            score = best["score"]
            # Match threshold from enroll_and_test.py is 0.90
            if score >= 0.90:
                verdict = "KNOWN"
                # Scale score 0.9->0.99 to 0->100%
                confidence = int(min(100, (score - 0.90) / 0.10 * 100))
                pid = best["person_id"]
                pname = best["name"]
            else:
                confidence = int(max(0, 100 - (score * 100)))

        persons_detected.append({
            "track_id": "track_1",
            "verdict": verdict,
            "confidence": confidence,
            "person_id": pid,
            "name": pname,
            "match_score": round(score, 4),
            "frames": total_frames,
            "thumbnail": thumb_b64
        })
            
    elapsed = int((time.time() - start) * 1000)

    return {
        "persons_detected": persons_detected,
        "total_frames": total_frames,
        "total_tracks": len(persons_detected),
        "processing_time_ms": elapsed
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/model/health")
async def health():
    return {
        "status": "ok",
        "enrolled": gallery.size(),
        "model_version": "DeepGaitV2 (Path A)",
    }

@app.get("/api/model/gallery")
async def list_gallery():
    # Frontend expects list of {person_id, name, embedded_count, etc.}
    res = []
    for pid, meta in gallery.meta.items():
        res.append({
            "person_id": pid,
            "name": meta["name"],
            "embedding_count": meta.get("video_count", 1)
        })
    return res

@app.delete("/api/model/gallery/{person_id}")
async def remove_from_gallery(person_id: str):
    gallery.delete(person_id)
    gallery.save()
    return {"ok": True}

@app.post("/api/model/register")
async def register_person(
    name: str = Form(...),
    person_id: str = Form(None),
    videos: List[UploadFile] = File(...),
):
    try:
        pid = person_id or str(uuid.uuid4())
        
        temp_paths = []
        for upload in videos:
            data = await upload.read()
            ext = os.path.splitext(upload.filename or ".mp4")[1] or ".mp4"
            temp_paths.append(save_tempfile(data, suffix=ext))
            
        import torch
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, process_registration, temp_paths, pid, name
        )
        
        for p in temp_paths:
            try: os.remove(p)
            except: pass
            
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/test")
async def test_video(video: UploadFile = File(...)):
    data = await video.read()
    ext = os.path.splitext(video.filename or ".mp4")[1] or ".mp4"
    path = save_tempfile(data, suffix=ext)
    
    import torch
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, process_test, path)
    
    try: os.remove(path)
    except: pass
    
    return report

if __name__ == "__main__":
    import uvicorn
    # Make sure we're in the right directory so paths like "weights/..." work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run("api_server:app", host="0.0.0.0", port=8005, reload=False)
