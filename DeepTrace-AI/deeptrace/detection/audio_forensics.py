"""Audio forensic analysis detecting synthetic voices, neural TTS and clone anomalies."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from deeptrace.config import Settings, get_settings
from deeptrace.detection.models import AudioSpectralModel
from deeptrace.logging_utils import get_logger
from deeptrace.preprocessing.audio_processor import AudioPreprocessor
from deeptrace.schemas import (
    ForensicArtifact,
    MediaForensicAnalysis,
    MediaType,
    ModelPrediction,
)

logger = get_logger(__name__)


class AudioForensicDetector:
    """Performs acoustic, spectral, and neural feature checks for synthesized speech."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.preprocessor = AudioPreprocessor(settings=self.settings)
        self.audio_model = AudioSpectralModel(settings=self.settings)

    def analyze(self, audio_path: Path | str) -> MediaForensicAnalysis:
        """Run acoustic and neural forensic checks on audio file."""
        audio, sr = self.preprocessor.load_audio(audio_path)
        spec = self.preprocessor.compute_spectrogram(audio, sr)
        acoustic = self.preprocessor.extract_acoustic_features(audio, sr)

        artifacts: List[ForensicArtifact] = []
        models: List[ModelPrediction] = []

        # 1. High-Frequency Vocoder Energy Cutoff
        hf_ratio = acoustic.get("hf_cutoff_energy_ratio", 0.05)
        hf_flagged = hf_ratio < 0.003
        artifacts.append(
            ForensicArtifact(
                name="Vocoder Frequency Rolloff Cutoff",
                indicator="neural_vocoder_ceiling",
                score=round(hf_ratio, 5),
                threshold=0.003,
                flagged=hf_flagged,
                description=(
                    "Synthetic neural vocoders (HiFi-GAN, WaveGlow) often truncate "
                    "acoustic frequencies sharply near Nyquist or 8kHz."
                ),
                metadata={"hf_ratio": round(hf_ratio, 6)},
            )
        )

        # 2. Spectral Flatness
        flatness = acoustic.get("spectral_flatness", 0.0)
        flatness_flagged = flatness > 0.12
        artifacts.append(
            ForensicArtifact(
                name="Spectral Flatness / Vocoder Buzz",
                indicator="vocoder_buzz_artifact",
                score=round(flatness, 4),
                threshold=0.12,
                flagged=flatness_flagged,
                description=(
                    "Measures tonal vs noise distribution. Unnaturally high flatness "
                    "in unvoiced speech is a hallmark of phase vocoder artifacts."
                ),
                metadata={"spectral_flatness": round(flatness, 5)},
            )
        )

        # 3. Dynamic Silence & Energy Distribution
        silence_ratio = acoustic.get("silence_ratio", 0.2)
        duration = acoustic.get("duration_s", 0.0)
        unnatural_silence = (silence_ratio < 0.02) and (duration > 3.0)
        artifacts.append(
            ForensicArtifact(
                name="Acoustic Pause & Energy Dynamics",
                indicator="synthetic_speech_pacing",
                score=round(silence_ratio, 3),
                threshold=0.02,
                flagged=unnatural_silence,
                description=(
                    "Human speech contains organic pauses, breaths, and natural decay. "
                    "Near-zero pause ratios indicate continuous synthetic audio synthesis."
                ),
                metadata={"silence_ratio": round(silence_ratio, 3), "duration_s": duration},
            )
        )

        # 4. Neural Acoustic Model Evaluation
        pred, _ = self.audio_model.evaluate_spectrogram(spec)
        models.append(pred)

        return MediaForensicAnalysis(
            media_type=MediaType.AUDIO,
            face_detected=False,
            face_count=0,
            artifacts=artifacts,
            models=models,
            audio_analysis=acoustic,
            audio_included=True,
        )
