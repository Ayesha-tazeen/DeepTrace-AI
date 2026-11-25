"""Image preprocessing and forensic feature extraction for DeepTrace AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deeptrace.config import Settings, get_settings
from deeptrace.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Preprocesses images and extracts forensic frequency & compression artifacts."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.yunet_detector = None
        self.face_cascade = None

        # 1. Try modern OpenCV FaceDetectorYN (YuNet)
        yunet_path = self.settings.models_dir / "face_detection_yunet.onnx"
        if hasattr(cv2, "FaceDetectorYN_create") and yunet_path.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(str(yunet_path), "", (320, 320))
                logger.info("Loaded YuNet face detector from %s", yunet_path)
            except Exception as exc:
                logger.warning("Could not initialize YuNet: %s", exc)

        # 2. Fallback to CascadeClassifier if available
        if self.yunet_detector is None and hasattr(cv2, "CascadeClassifier"):
            local_model = self.settings.models_dir / "haarcascade_frontalface_default.xml"
            cascade_path = None
            if local_model.exists():
                cascade_path = str(local_model)
            elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cv2_cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cv2_cascade):
                    cascade_path = cv2_cascade

            if cascade_path and os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Loaded face cascade from %s", cascade_path)

    def load_image(self, path: Path | str) -> np.ndarray:
        """Load an image safely supporting Unicode Windows paths."""
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image at {path}")
        return img

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in BGR image. Returns list of (x, y, w, h)."""
        h_img, w_img = image.shape[:2]

        # Use YuNet if available
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w_img, h_img))
                _, detected = self.yunet_detector.detect(image)
                if detected is not None and len(detected) > 0:
                    faces = []
                    for face in detected:
                        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        # Keep within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w_img - x, w)
                        h = min(h_img - y, h)
                        if w > 20 and h > 20:
                            faces.append((x, y, w, h))
                    return faces
            except Exception as exc:
                logger.warning("YuNet detection error: %s", exc)

        # Use Haar cascade if available
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                return [tuple(map(int, f)) for f in faces]
            except Exception as exc:
                logger.warning("Haar cascade error: %s", exc)

        return []

    def extract_face_crops(
        self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], target_size: int = 224
    ) -> List[np.ndarray]:
        """Crop and resize faces to target size."""
        crops = []
        h_img, w_img = image.shape[:2]
        for x, y, w, h in faces:
            # Add 15% margin around face
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
                crops.append(resized)
        return crops

    def compute_ela(
        self, image: np.ndarray, quality: int = 90, scale: int = 15
    ) -> Tuple[np.ndarray, float]:
        """Compute Error Level Analysis (ELA) map and average discrepancy score.

        Detects compression artifact discrepancies (common when parts of an image
        were generated or spliced with different compression histories).
        """
        # Save to memory as JPEG at fixed quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", image, encode_param)
        if not success:
            return np.zeros_like(image), 0.0

        recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        # Compute absolute difference
        diff = cv2.absdiff(image, recompressed)
        diff_scaled = cv2.scaleAdd(diff, float(scale), np.zeros_like(diff))
        ela_score = float(np.mean(diff))
        return diff_scaled, ela_score

    def compute_fft_frequency_spectrum(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze 2D Fourier spectrum to detect high-frequency artifacts.

        GANs and diffusion models often introduce periodic grid patterns
        or anomalous high-frequency energy distributions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Center crop to square power-of-two or standard size
        size = min(h, w, 512)
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        cropped = gray[start_y : start_y + size, start_x : start_x + size]

        # 2D FFT
        dft = np.fft.fft2(cropped)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-6)

        # Calculate high-frequency energy ratio
        center_y, center_x = size // 2, size // 2
        radius_low = size // 8
        radius_mid = size // 4

        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        low_mask = dist_from_center <= radius_low
        high_mask = dist_from_center > radius_mid

        low_energy = float(np.mean(magnitude_spectrum[low_mask]))
        high_energy = float(np.mean(magnitude_spectrum[high_mask]))
        hf_ratio = float(high_energy / (low_energy + 1e-6))

        # Check for azimuth symmetry anomaly (periodic spikes in Fourier domain)
        angles = np.arctan2(y - center_y, x - center_x)
        # Radial profile standard deviation
        radial_var = float(np.var(magnitude_spectrum[high_mask]))

        return {
            "high_frequency_ratio": round(hf_ratio, 4),
            "radial_variance": round(radial_var, 4),
            "low_energy_mean": round(low_energy, 4),
            "high_energy_mean": round(high_energy, 4),
            "spectral_dim": size,
        }

    def compute_noise_consistency(
        self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]
    ) -> Dict[str, float]:
        """Compare Laplacian noise variance in facial ROI vs whole image / background."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        global_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if not faces:
            return {
                "global_noise_variance": round(global_var, 3),
                "face_noise_variance": round(global_var, 3),
                "noise_discrepancy_ratio": 1.0,
            }

        # Face variance
        face_vars = []
        for x, y, w, h in faces:
            face_gray = gray[y : y + h, x : x + w]
            if face_gray.size > 0:
                face_vars.append(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()))

        avg_face_var = float(np.mean(face_vars)) if face_vars else global_var
        ratio = float(avg_face_var / (global_var + 1e-6))

        return {
            "global_noise_variance": round(global_var, 3),
            "face_noise_variance": round(avg_face_var, 3),
            "noise_discrepancy_ratio": round(ratio, 3),
        }
