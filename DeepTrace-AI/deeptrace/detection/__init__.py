"""DeepTrace detection package."""

from deeptrace.detection.audio_forensics import AudioForensicDetector
from deeptrace.detection.image_forensics import ImageForensicDetector
from deeptrace.detection.models import AudioSpectralModel, VisionDeepfakeModel
from deeptrace.detection.video_forensics import VideoForensicDetector

__all__ = [
    "VisionDeepfakeModel",
    "AudioSpectralModel",
    "ImageForensicDetector",
    "VideoForensicDetector",
    "AudioForensicDetector",
]
