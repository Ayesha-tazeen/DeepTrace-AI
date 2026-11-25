"""Video preprocessing, frame sampling, face tracking and audio extraction."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from deeptrace.config import Settings, get_settings
from deeptrace.logging_utils import get_logger
from deeptrace.preprocessing.image_processor import ImagePreprocessor

logger = get_logger(__name__)


class VideoPreprocessor:
    """Samples frames, analyzes inter-frame temporal consistency, and extracts audio."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.image_processor = ImagePreprocessor(settings=self.settings)

    def extract_audio_track(self, video_path: Path | str) -> Optional[Path]:
        """Extract audio from video to a temporary WAV file using imageio-ffmpeg or ffmpeg."""
        self.settings.ensure_directories()
        video_path = Path(video_path)
        output_wav = self.settings.temp_dir / f"{video_path.stem}_audio.wav"

        # Try imageio_ffmpeg first if available
        ffmpeg_bin = None
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"

        try:
            cmd = [
                ffmpeg_bin,
                "-y",  # overwrite
                "-i", str(video_path),
                "-vn",  # no video
                "-acodec", "pcm_s16le",
                "-ar", str(self.settings.audio_sample_rate),
                "-ac", "1",  # mono
                str(output_wav),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
            if result.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 1000:
                logger.info("Extracted audio track to %s", output_wav)
                return output_wav
            else:
                logger.debug("No valid audio track extracted from %s", video_path.name)
        except Exception as exc:
            logger.debug("Audio extraction skipped or failed: %s", exc)

        return None

    def sample_frames(
        self, video_path: Path | str
    ) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
        """Sample frames uniformly according to max_sampled_fps and processing_max_seconds.

        Returns: (frames_bgr, timestamps_s, video_meta)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        try:
            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total_frames / (native_fps if native_fps > 0 else 25.0)

            # Cap duration to processing_max_seconds
            effective_duration = min(duration, self.settings.processing_max_seconds)
            effective_frames = int(effective_duration * native_fps)

            # Determine sampling interval
            step = max(1, int(round(native_fps / self.settings.max_sampled_fps)))

            sampled_frames: List[np.ndarray] = []
            timestamps: List[float] = []

            curr_frame_idx = 0
            while curr_frame_idx < effective_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Downscale if larger than image_max_side
                h, w = frame.shape[:2]
                if max(h, w) > self.settings.image_max_side:
                    scale = self.settings.image_max_side / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                timestamp = curr_frame_idx / native_fps
                sampled_frames.append(frame)
                timestamps.append(round(timestamp, 3))
                curr_frame_idx += step

            meta = {
                "native_fps": native_fps,
                "total_frames": total_frames,
                "duration_s": round(duration, 3),
                "analyzed_duration_s": round(effective_duration, 3),
                "sampled_frame_count": len(sampled_frames),
                "resolution": f"{width}x{height}",
            }
            return sampled_frames, timestamps, meta
        finally:
            cap.release()

    def analyze_temporal_consistency(
        self, frames: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Measure inter-frame difference and optical flow variance.

        Deepfake videos generated frame-by-frame without temporal smoothing
        frequently suffer from high-frequency pixel jitter and optical flow anomalies.
        """
        if len(frames) < 2:
            return {"temporal_jitter_score": 0.0, "mean_frame_diff": 0.0}

        diffs = []
        flow_magnitudes = []

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (256, 256))

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.resize(curr_gray, (256, 256))

            # Absolute difference
            diff = float(np.mean(cv2.absdiff(curr_gray, prev_gray)))
            diffs.append(diff)

            # Optical flow (Farneback)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_magnitudes.append(float(np.mean(mag)))
            except Exception:
                pass

            prev_gray = curr_gray

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        diff_var = float(np.var(diffs)) if diffs else 0.0
        mean_flow = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        flow_var = float(np.var(flow_magnitudes)) if flow_magnitudes else 0.0

        # High flow variance relative to mean motion is an indicator of jitter
        jitter_score = float(flow_var / (mean_flow + 1.0))

        return {
            "mean_frame_diff": round(mean_diff, 3),
            "diff_variance": round(diff_var, 3),
            "mean_optical_flow": round(mean_flow, 3),
            "flow_variance": round(flow_var, 3),
            "temporal_jitter_score": round(jitter_score, 4),
        }
