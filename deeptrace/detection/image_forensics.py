"""Image forensic analysis module combining AI features with measurable forensic indicators."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from deeptrace.config import Settings, get_settings
from deeptrace.detection.models import VisionDeepfakeModel
from deeptrace.logging_utils import get_logger
from deeptrace.preprocessing.image_processor import ImagePreprocessor
from deeptrace.schemas import (
    ForensicArtifact,
    MediaForensicAnalysis,
    MediaType,
    ModelPrediction,
)

logger = get_logger(__name__)


class ImageForensicDetector:
    """Analyzes images for deepfake markers, compression inconsistencies, and frequency anomalies."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.preprocessor = ImagePreprocessor(settings=self.settings)
        self.vision_model = VisionDeepfakeModel(settings=self.settings)

    def analyze(self, image_path: Path | str) -> MediaForensicAnalysis:
        """Execute full forensic analysis on an image."""
        img = self.preprocessor.load_image(image_path)
        faces = self.preprocessor.detect_faces(img)
        face_detected = len(faces) > 0
        face_boxes = [list(f) for f in faces]

        artifacts: List[ForensicArtifact] = []
        models: List[ModelPrediction] = []

        # 1. Error Level Analysis (ELA)
        _, ela_score = self.preprocessor.compute_ela(img, quality=90)
        ela_flagged = ela_score > 9.5
        artifacts.append(
            ForensicArtifact(
                name="Error Level Analysis (ELA)",
                indicator="compression_discrepancy",
                score=round(ela_score, 2),
                threshold=9.5,
                flagged=ela_flagged,
                description=(
                    "Measures compression variance across regions. High scores suggest "
                    "spliced or re-compressed synthetic artifacts."
                ),
                metadata={"ela_mean_residual": round(ela_score, 3)},
            )
        )

        # 2. 2D Fourier Frequency Spectrum
        freq_spectrum = self.preprocessor.compute_fft_frequency_spectrum(img)
        hf_ratio = freq_spectrum.get("high_frequency_ratio", 0.0)
        rad_var = freq_spectrum.get("radial_variance", 0.0)

        freq_flagged = (hf_ratio > 0.45) or (rad_var > 130.0)
        artifacts.append(
            ForensicArtifact(
                name="Frequency Domain FFT Spectrum",
                indicator="periodic_grid_artifacts",
                score=round(hf_ratio, 4),
                threshold=0.45,
                flagged=freq_flagged,
                description=(
                    "Analyzes high-frequency Fourier energy. Periodic spikes or unnaturally "
                    "elevated high-frequency ratios indicate GAN or diffusion generation."
                ),
                metadata=freq_spectrum,
            )
        )

        # 3. Spatial Noise Consistency (Facial ROI vs Background)
        noise_analysis = self.preprocessor.compute_noise_consistency(img, faces)
        noise_ratio = noise_analysis.get("noise_discrepancy_ratio", 1.0)
        noise_flagged = (noise_ratio < 0.35) or (noise_ratio > 2.8)
        if face_detected:
            artifacts.append(
                ForensicArtifact(
                    name="Facial ROI Noise Consistency",
                    indicator="boundary_blending_anomaly",
                    score=round(noise_ratio, 3),
                    threshold=2.8,
                    flagged=noise_flagged,
                    description=(
                        "Compares Laplacian noise variance in face region against global context. "
                        "Severe disparity indicates artificial face-swapping or smoothing."
                    ),
                    metadata=noise_analysis,
                )
            )

        # 4. Neural AI Evaluation
        if face_detected:
            face_crops = self.preprocessor.extract_face_crops(img, faces)
            pred, _ = self.vision_model.evaluate_crops(face_crops)
            models.append(pred)
        else:
            # Whole image crop if no face detected
            h, w = img.shape[:2]
            dim = min(h, w)
            center_crop = img[(h - dim) // 2 : (h + dim) // 2, (w - dim) // 2 : (w + dim) // 2]
            pred, _ = self.vision_model.evaluate_crops([center_crop])
            models.append(pred)

        compression_info = {
            "ela_residual": round(ela_score, 3),
            "ela_flagged": ela_flagged,
        }

        return MediaForensicAnalysis(
            media_type=MediaType.IMAGE,
            face_detected=face_detected,
            face_count=len(faces),
            face_bounding_boxes=face_boxes,
            artifacts=artifacts,
            models=models,
            frequency_analysis=freq_spectrum,
            compression_analysis=compression_info,
            spatial_consistency=noise_analysis,
        )
