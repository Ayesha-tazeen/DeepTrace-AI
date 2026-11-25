"""Unit tests for image, video, and audio preprocessing."""

import numpy as np
import pytest

from deeptrace.preprocessing.audio_processor import AudioPreprocessor
from deeptrace.preprocessing.image_processor import ImagePreprocessor
from deeptrace.preprocessing.video_processor import VideoPreprocessor


def test_image_preprocessing(jpg_file, settings):
    proc = ImagePreprocessor(settings)
    img = proc.load_image(jpg_file)
    assert isinstance(img, np.ndarray)

    ela_map, ela_score = proc.compute_ela(img)
    assert ela_map.shape == img.shape
    assert ela_score >= 0.0

    fft_info = proc.compute_fft_frequency_spectrum(img)
    assert "high_frequency_ratio" in fft_info
    assert "radial_variance" in fft_info

    noise_info = proc.compute_noise_consistency(img, faces=[])
    assert "global_noise_variance" in noise_info


def test_audio_preprocessing(wav_file, settings):
    proc = AudioPreprocessor(settings)
    audio, sr = proc.load_audio(wav_file)
    assert len(audio) > 0
    assert sr == settings.audio_sample_rate

    spec = proc.compute_spectrogram(audio, sr)
    assert spec.ndim == 2

    features = proc.extract_acoustic_features(audio, sr)
    assert "mean_energy_rms" in features
    assert "zero_crossing_rate" in features
    assert "spectral_flatness" in features


def test_video_preprocessing(mp4_file, settings):
    proc = VideoPreprocessor(settings)
    frames, timestamps, meta = proc.sample_frames(mp4_file)
    assert len(frames) > 0
    assert len(frames) == len(timestamps)
    assert meta["total_frames"] > 0

    temporal_info = proc.analyze_temporal_consistency(frames)
    assert "temporal_jitter_score" in temporal_info
