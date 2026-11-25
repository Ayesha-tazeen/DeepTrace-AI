"""Video forensic detector integrating multi-frame temporal consistency, face tracking and audio forensics."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from deeptrace.config import Settings, get_settings
from deeptrace.detection.audio_forensics import AudioForensicDetector
from deeptrace.detection.image_forensics import ImageForensicDetector
from deeptrace.logging_utils import get_logger
from deeptrace.preprocessing.video_processor import VideoPreprocessor
from deeptrace.schemas import (
    ForensicArtifact,
    MediaForensicAnalysis,
    MediaType,
    ModelPrediction,
)

logger = get_logger(__name__)


class VideoForensicDetector:
    """Analyzes video streams for facial deepfakes, temporal flickering, and extracted audio forensics."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.video_processor = VideoPreprocessor(settings=self.settings)
        self.image_detector = ImageForensicDetector(settings=self.settings)
        self.audio_detector = AudioForensicDetector(settings=self.settings)

    def analyze(self, video_path: Path | str) -> MediaForensicAnalysis:
        """Run full multimodal forensic analysis on video."""
        video_path = Path(video_path)
        frames, timestamps, meta = self.video_processor.sample_frames(video_path)

        artifacts: List[ForensicArtifact] = []
        models: List[ModelPrediction] = []

        # 1. Temporal consistency analysis
        temporal_info = self.video_processor.analyze_temporal_consistency(frames)
        jitter_score = temporal_info.get("temporal_jitter_score", 0.0)
        jitter_flagged = jitter_score > 0.12
        artifacts.append(
            ForensicArtifact(
                name="Inter-Frame Temporal Flicker",
                indicator="temporal_inconsistency",
                score=round(jitter_score, 4),
                threshold=0.12,
                flagged=jitter_flagged,
                description=(
                    "Measures frame-to-frame optical flow variance. Deepfakes generated without "
                    "temporal constraints suffer from facial and boundary jitter."
                ),
                metadata=temporal_info,
            )
        )

        # 2. Sample frame facial & ELA analysis
        face_detected_any = False
        face_counts = []
        frame_ela_scores = []
        frame_predictions: List[ModelPrediction] = []

        # Analyze a subset of sampled frames (up to 8 frames evenly spaced)
        num_frames_to_test = min(len(frames), 8)
        indices = np.linspace(0, len(frames) - 1, num_frames_to_test, dtype=int) if frames else []

        for idx in indices:
            frame = frames[idx]
            faces = self.video_processor.image_processor.detect_faces(frame)
            if faces:
                face_detected_any = True
                face_counts.append(len(faces))

            _, ela_val = self.video_processor.image_processor.compute_ela(frame)
            frame_ela_scores.append(ela_val)

            # Vision model evaluation on frame
            if faces:
                crops = self.video_processor.image_processor.extract_face_crops(frame, faces)
                pred, _ = self.image_detector.vision_model.evaluate_crops(crops)
                frame_predictions.append(pred)

        avg_ela = float(np.mean(frame_ela_scores)) if frame_ela_scores else 0.0
        ela_flagged = avg_ela > 9.5
        artifacts.append(
            ForensicArtifact(
                name="Multi-Frame ELA Compression Discrepancy",
                indicator="video_compression_residual",
                score=round(avg_ela, 2),
                threshold=9.5,
                flagged=ela_flagged,
                description="Average Error Level Analysis across sampled keyframes.",
                metadata={"average_ela": round(avg_ela, 3), "keyframes_evaluated": len(indices)},
            )
        )

        if frame_predictions:
            models.extend(frame_predictions[:2])  # Include representative model outputs
        else:
            # Fallback model prediction from image detector
            if frames:
                pred, _ = self.image_detector.vision_model.evaluate_crops([frames[0]])
                models.append(pred)

        # 3. Audio extraction & analysis
        audio_included = False
        audio_analysis_data = None
        audio_wav = self.video_processor.extract_audio_track(video_path)
        if audio_wav and audio_wav.exists():
            try:
                audio_res = self.audio_detector.analyze(audio_wav)
                audio_included = True
                audio_analysis_data = audio_res.audio_analysis
                # Append audio artifacts with prefix
                for art in audio_res.artifacts:
                    art.name = f"[Audio Track] {art.name}"
                    artifacts.append(art)
                models.extend(audio_res.models)
            except Exception as exc:
                logger.warning("Error analyzing extracted audio track: %s", exc)

        return MediaForensicAnalysis(
            media_type=MediaType.VIDEO,
            face_detected=face_detected_any,
            face_count=int(max(face_counts)) if face_counts else 0,
            frames_analyzed=len(frames),
            artifacts=artifacts,
            models=models,
            temporal_consistency=temporal_info,
            audio_included=audio_included,
            audio_analysis=audio_analysis_data,
        )
