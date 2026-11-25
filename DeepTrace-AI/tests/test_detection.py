"""Unit tests for genuine AI models and forensic detectors."""

from deeptrace.detection.audio_forensics import AudioForensicDetector
from deeptrace.detection.image_forensics import ImageForensicDetector
from deeptrace.detection.models import AudioSpectralModel, VisionDeepfakeModel
from deeptrace.detection.video_forensics import VideoForensicDetector
from deeptrace.schemas import MediaType


def test_vision_model_and_image_detector(jpg_file, settings):
    model = VisionDeepfakeModel(settings)
    assert model.model is not None
    assert "feature_mode" in model.weights_status or "loaded" in model.weights_status

    detector = ImageForensicDetector(settings)
    analysis = detector.analyze(jpg_file)
    assert analysis.media_type is MediaType.IMAGE
    assert len(analysis.artifacts) >= 2
    assert len(analysis.models) >= 1
    assert analysis.models[0].model_name == "EfficientNet-B0"


def test_audio_model_and_audio_detector(wav_file, settings):
    model = AudioSpectralModel(settings)
    assert model.conv_net is not None

    detector = AudioForensicDetector(settings)
    analysis = detector.analyze(wav_file)
    assert analysis.media_type is MediaType.AUDIO
    assert len(analysis.artifacts) >= 3
    assert len(analysis.models) >= 1


def test_video_detector(mp4_file, settings):
    detector = VideoForensicDetector(settings)
    analysis = detector.analyze(mp4_file)
    assert analysis.media_type is MediaType.VIDEO
    assert analysis.frames_analyzed is not None
    assert len(analysis.artifacts) >= 2
