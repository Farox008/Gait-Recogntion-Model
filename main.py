"""
main.py — GaitPipeline

Full pipeline entry point.  Wires together all components and can be run
either on a video file (for testing) or a live RTSP stream.

Usage:
  python main.py run   --source rtsp://cam1/... --camera-id cam1
  python main.py enroll --clip person.mp4 --person-id P001 --name "Alice"
  python main.py test  --source test_video.mp4 --camera-id test
"""
import argparse
import logging
import os
import sys
import time
from queue import Queue

import httpx
import yaml

# ── Absolute imports from project root ────────────────────────────────────────
from pipeline.stream_reader      import StreamReader
from pipeline.detector           import PersonDetector
from pipeline.tracker            import GaitTracker
from pipeline.pose_estimator     import PoseEstimator
from pipeline.silhouette_extractor import SilhouetteExtractor
from pipeline.denoiser           import GaitDenoiser
from encoders.skeleton_encoder   import SkeletonEncoder
from encoders.silhouette_encoder import SilhouetteEncoder
from encoders.fusion_module      import GaitFusionModule, GaitEmbedder
from gallery.gallery             import GaitGallery
from gallery.embedder            import EnrollmentEmbedder
from identification.identifier   import GaitIdentifier
from identification.verifier     import UnknownVerifier
from identification.alert        import AlertBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gait_pipeline")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class GaitPipeline:

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        cfg = _load_config(config_path)
        self.cfg       = cfg
        self.model_cfg = cfg["model"]
        self.pipe_cfg  = cfg["pipeline"]
        self.id_cfg    = cfg["identification"]
        self.backend   = cfg["backend"]

        device = self.model_cfg.get("device", "cpu")

        logger.info("Initialising pipeline components …")

        # ── Detection / tracking ────────────────────────────────────────────
        self.detector = PersonDetector(
            model="yolov8n.pt",
            conf=0.4,
            device=device,
            detect_every=self.pipe_cfg.get("detect_every_n_frames", 3),
        )
        self.tracker = GaitTracker(max_age=30)

        # ── Pose & silhouette streams ────────────────────────────────────────
        self.pose_estimator    = PoseEstimator(model="yolov8n-pose.pt", device=device)
        self.sil_extractor     = SilhouetteExtractor()
        self.denoiser          = GaitDenoiser()

        # ── Encoders ────────────────────────────────────────────────────────
        skel_weights = self.model_cfg.get("skeleton_weights", "weights/skeletongait++.pt")
        sil_weights  = self.model_cfg.get("silhouette_weights", "weights/deepgaitv2.pt")
        fusion_weights = self.model_cfg.get("fusion_weights", "weights/fusion.pt")

        self.skel_encoder = SkeletonEncoder(skel_weights, device=device)
        self.sil_encoder  = SilhouetteEncoder(sil_weights, device=device)
        fusion_module     = GaitFusionModule(embed_dim=256, num_heads=8)
        self.embedder     = GaitEmbedder(
            self.skel_encoder, self.sil_encoder, fusion_module,
            device=device, fusion_weights_path=fusion_weights
        )

        # ── Gallery ─────────────────────────────────────────────────────────
        self.gallery = GaitGallery(
            index_path=self.model_cfg.get("gallery_index", "weights/gallery.faiss"),
            meta_path =self.model_cfg.get("gallery_meta",  "weights/gallery_meta.json"),
        )
        self.gallery.load()

        # ── Identification ──────────────────────────────────────────────────
        self.identifier    = GaitIdentifier(self.gallery, self.embedder, self.id_cfg)
        self.verifier      = UnknownVerifier(
            required_consecutive=self.id_cfg.get("consecutive_unknowns_required", 3),
            cooldown_seconds    =self.id_cfg.get("cooldown_seconds", 300),
        )
        self.alert_builder = AlertBuilder()

        # ── Enrollment helper (reuses all components) ───────────────────────
        self.enrollment_embedder = EnrollmentEmbedder(
            detector=self.detector,
            tracker =self.tracker,
            pose_estimator=self.pose_estimator,
            silhouette_extractor=self.sil_extractor,
            denoiser=self.denoiser,
            gait_embedder=self.embedder,
        )

        logger.info("Pipeline ready.")

    # ── Public interface ────────────────────────────────────────────────────

    def run(self, source: str, camera_id: str):
        """
        Process an RTSP stream or video file.
        Blocks until the stream ends or KeyboardInterrupt.
        """
        frame_queue: Queue = Queue(maxsize=256)
        reader = StreamReader(source, camera_id, frame_queue)
        reader.start()

        # Warm-up: collect a few frames for background model
        warmup_frames = []
        logger.info("Warming up background model …")
        while len(warmup_frames) < 30:
            try:
                _, frame, _ = frame_queue.get(timeout=5)
                warmup_frames.append(frame)
            except Exception:
                logger.warning("Timeout waiting for warmup frames.")
                break
        if warmup_frames:
            self.sil_extractor.fit_background(warmup_frames)

        logger.info(f"Processing stream from {source} …")
        try:
            while True:
                try:
                    _, frame, _ = frame_queue.get(timeout=5)
                except Exception:
                    continue

                detections = self.detector.detect(frame)
                tracks     = self.tracker.update(detections, frame)

                for track in tracks:
                    if not track.READY:
                        continue

                    pairs   = track.frame_bbox_pairs()
                    hm_seq  = self.pose_estimator.build_sequence(pairs)
                    sil_seq = self.sil_extractor.extract_sequence(pairs)
                    sil_seq = self.denoiser.denoise_sequence(sil_seq)

                    result = self.identifier.identify(hm_seq, sil_seq)

                    logger.debug(
                        f"  Track {track.track_id}: {result.verdict} "
                        f"(score={result.confidence_score:.3f})"
                    )

                    if self.verifier.check(str(track.track_id), result):
                        payload = self.alert_builder.build(
                            camera_id=camera_id,
                            track_id=str(track.track_id),
                            result=result,
                            snapshot_frame=track.best_frame,
                            gait_clip_frames=list(track.frames),
                        )
                        self.post_alert(payload)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
        finally:
            reader.stop()

    def enroll_person(self, person_id: str, display_name: str, clip_path: str):
        """Enroll a new known person from a video clip."""
        logger.info(f"Enrolling '{display_name}' from {clip_path} …")
        embedding = self.enrollment_embedder.embed_from_clip(clip_path)
        self.gallery.enroll(person_id, display_name, embedding)
        self.gallery.save()
        logger.info(f"Enrolled '{display_name}' ({person_id}). Gallery size: {self.gallery.size()}")

    def post_alert(self, payload: dict):
        """POST alert payload to the FastAPI backend."""
        url = f"{self.backend['url'].rstrip('/')}/api/alerts/ingest"
        headers = {"X-Engine-Secret": self.backend.get("secret_key", "")}
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            logger.info(f"Alert posted → {resp.json()}")
        except Exception as exc:
            logger.error(f"Failed to post alert: {exc}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gait Recognition Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # run
    p_run = sub.add_parser("run", help="Process RTSP stream or video file")
    p_run.add_argument("--source",    required=True, help="RTSP URL or path to video file")
    p_run.add_argument("--camera-id", default="cam0", help="Camera identifier string")

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll new person from a video clip")
    p_enroll.add_argument("--clip",      required=True, help="Path to enrollment video clip")
    p_enroll.add_argument("--person-id", required=True, help="Unique person ID (e.g. EMP001)")
    p_enroll.add_argument("--name",      required=True, help="Full display name")

    # test
    p_test = sub.add_parser("test", help="Quick smoke-test on a video file")
    p_test.add_argument("--source",    required=True, help="Path to test video file")
    p_test.add_argument("--camera-id", default="test")

    parser.add_argument("--config", default="configs/model_config.yaml")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    pipeline = GaitPipeline(config_path=args.config)

    if args.cmd == "run":
        pipeline.run(args.source, args.camera_id)
    elif args.cmd == "enroll":
        pipeline.enroll_person(args.person_id, args.name, args.clip)
    elif args.cmd == "test":
        logger.info("Running smoke-test — watching for alerts …")
        pipeline.run(args.source, args.camera_id)


if __name__ == "__main__":
    main()
