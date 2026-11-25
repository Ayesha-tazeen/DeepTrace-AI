"""Shared pytest fixtures: tiny synthetic media files built at test time."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from deeptrace.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs")


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture()
def jpg_file(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    rate, seconds = 16000, 2
    t = np.linspace(0, seconds, rate * seconds, endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


@pytest.fixture()
def mp4_file(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write mp4 files")
    rng = np.random.default_rng(2)
    for _ in range(20):
        writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()
    return path
