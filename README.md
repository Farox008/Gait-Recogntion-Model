# Gait Recognition Model

A high-performance, multi-modal Gait Recognition system that identifies individuals by the way they walk. This system combines **Skeleton-based** and **Silhouette-based** features to create a unique "gait fingerprint," and supports viewpoint-specific recognition for enhanced accuracy in CCTV environments.

## 🚀 Key Features

- **Multi-Modal Fusion**: Combines YOLOv8-Pose (skeletons) and YOLOv8-Seg (silhouettes) for robust identification.
- **DeepGaitV2 & SkeletonGait++**: Integrates state-of-the-art encoders for feature extraction.
- **Camera-Specific Recognition**: Supports separate galleries for different camera IDs to handle viewpoint variations.
- **Interactive Runner**: A user-friendly CLI menu for training, testing, and batch processing.
- **Live RTSP Support**: Capable of processing real-time streams and sending alerts to a backend.
- **FAISS Powered**: Instant identification using Facebook AI Similarity Search.

---

## 🛠 Project Architecture

The system is built with a modular pipeline:

1.  **Detection & Tracking**: YOLOv8 detects people, and a tracker maintains identity across frames.
2.  **Feature Extraction**:
    *   `PoseEstimator`: Extracts 2D skeletal joints.
    *   `SilhouetteExtractor`: Generates binary silhouettes.
3.  **Encoding**:
    *   `SkeletonEncoder`: Process skeletons via spatio-temporal layers.
    *   `SilhouetteEncoder`: Process silhouettes using DeepGaitV2 architecture.
    *   `FusionModule`: Merges modalities into a single high-dimensional embedding.
4.  **Gallery Management**: FAISS-backed index for ultra-fast similarity search and persistence.
5.  **Alerting**: Sends JSON payloads with identification results and snapshots to a configured API.

---

## 🏃 Getting Started

### 1. Prerequisites
- Python 3.10+
- CUDA-compatible GPU (Highly recommended for real-time performance)
- Git LFS (Required to download the pre-trained weights)

### 2. Installation
```bash
git clone https://github.com/Farox008/Gait-Recogntion-Model.git
cd Gait-Recogntion-Model

# Install Git LFS to pull the large weight files
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Model
The easiest way to use the system is via the **Interactive Runner**:
```bash
python run.py
```
This script provides 6 options:
1.  **Train (Enroll) only**: Process your VODs directory to register known people.
2.  **Run (Test) only**: Run identification on your test data.
3.  **Both**: Full pipeline (Enroll then Identify).
4.  **Custom Directory Test**: Point the model at any folder containing video data.
5.  **Single Video Test**: Quickly identify a person in one specific video file.
6.  **Exit**

---

## 📷 Viewpoint-Specific Recognition

Gait can look different depending on the camera angle. To maximize accuracy, you can use the **Camera ID** feature:
- When prompted in `run.py`, enter a unique ID for your camera (e.g., `lobby_east`).
- The system will create/load a specific gallery for that camera (`weights/gallery_lobby_east.faiss`).
- This restricts identification to the specific perspective of that camera, significantly reducing false positives.

---

## 📦 Model Weights
This repository includes the following pre-trained models via Git LFS:
- `weights/deepgaitv2.pt`: Silhouette-based encoder.
- `weights/skeletongait++.pt`: Skeleton-based encoder.
- `weights/fusion.pt`: Multi-modal fusion weights.
- `yolov8n-pose.pt`, `yolov8n-seg.pt`: Vision backbone models.

---

## 📝 Recent Improvements & Changes
- ✅ **New Interactive Runner**: Unified all scripts into a single `run.py` interface.
- ✅ **Custom Test Logic**: Added support for testing single files and custom directories.
- ✅ **Camera Indexing**: Implemented persistent camera-specific memory for the gallery.
- ✅ **Segmentation Upgrade**: Switched to YOLOv8-Seg for much sharper silhouette extraction compared to traditional background subtraction.
