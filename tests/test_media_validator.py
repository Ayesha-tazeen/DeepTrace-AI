"""Unit tests for deeptrace.ingestion.media_validator."""

from pathlib import Path

import pytest

from deeptrace.exceptions import MediaValidationError
from deeptrace.ingestion import ingest_media, validate_media
from deeptrace.schemas import MediaType


# ---------------------------- valid files -------------------------------
def test_valid_png(png_file, settings):
    result = validate_media(png_file, settings)
    assert result.is_valid, result.errors
    assert result.media_type is MediaType.IMAGE
    assert (result.width, result.height) == (128, 96)
    assert result.mime_type == "image/png"
    assert len(result.sha256) == 64


def test_valid_jpg(jpg_file, settings):
    result = validate_media(jpg_file, settings)
    assert result.is_valid, result.errors
    assert result.mime_type == "image/jpeg"


def test_valid_wav(wav_file, settings):
    result = validate_media(wav_file, settings)
    assert result.is_valid, result.errors
    assert result.media_type is MediaType.AUDIO
    assert result.sample_rate == 16000
    assert result.duration_s == pytest.approx(2.0, abs=0.05)


def test_valid_mp4(mp4_file, settings):
    result = validate_media(mp4_file, settings)
    assert result.is_valid, result.errors
    assert result.media_type is MediaType.VIDEO
    assert result.fps == pytest.approx(10.0, abs=0.5)
    assert result.duration_s == pytest.approx(2.0, abs=0.3)


# --------------------------- invalid files ------------------------------
def test_missing_file(tmp_path, settings):
    result = validate_media(tmp_path / "nope.png", settings)
    assert not result.is_valid


def test_directory_rejected(tmp_path, settings):
    assert not validate_media(tmp_path, settings).is_valid


def test_unsupported_extension(tmp_path, settings):
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    result = validate_media(f, settings)
    assert not result.is_valid
    assert "Unsupported" in result.errors[0]


def test_text_renamed_to_png_rejected(tmp_path, settings):
    f = tmp_path / "fake.png"
    f.write_text("this is not an image")
    result = validate_media(f, settings)
    assert not result.is_valid
    assert "does not match" in result.errors[0]


def test_empty_file(tmp_path, settings):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    assert "empty" in validate_media(f, settings).errors[0].lower()


def test_path_traversal_rejected(png_file, settings):
    sneaky = str(png_file.parent / ".." / png_file.parent.name / png_file.name)
    result = validate_media(sneaky, settings)
    assert not result.is_valid
    assert any("traversal" in e.lower() for e in result.errors)


def test_file_outside_allowed_root(png_file, settings, tmp_path):
    other_root = tmp_path / "somewhere_else"
    other_root.mkdir()
    result = validate_media(png_file, settings, allowed_root=other_root)
    assert not result.is_valid


def test_oversize_file(png_file, settings):
    settings.max_image_size_mb = 0.0001  # ~100 bytes
    result = validate_media(png_file, settings)
    assert not result.is_valid
    assert "limit" in result.errors[0]


def test_audio_too_long(wav_file, settings):
    settings.max_audio_duration_s = 1.0
    result = validate_media(wav_file, settings)
    assert not result.is_valid
    assert "limit" in result.errors[-1]


def test_video_too_long(mp4_file, settings):
    settings.max_video_duration_s = 1.0
    assert not validate_media(mp4_file, settings).is_valid


# ----------------------------- corrupted --------------------------------
def test_truncated_png_rejected(png_file, settings, tmp_path):
    data = png_file.read_bytes()
    bad = tmp_path / "truncated.png"
    bad.write_bytes(data[: len(data) // 2])
    assert not validate_media(bad, settings).is_valid


def test_png_signature_with_garbage_rejected(tmp_path, settings):
    bad = tmp_path / "garbage.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert not validate_media(bad, settings).is_valid


def test_corrupted_wav_rejected(wav_file, settings, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(wav_file.read_bytes()[:30])  # header cut off
    assert not validate_media(bad, settings).is_valid


def test_corrupted_mp4_rejected(mp4_file, settings, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(mp4_file.read_bytes()[:40])
    assert not validate_media(bad, settings).is_valid


def test_fake_mp3_rejected(tmp_path, settings):
    bad = tmp_path / "fake.mp3"
    bad.write_bytes(b"ID3" + b"\x00" * 100)
    assert not validate_media(bad, settings).is_valid


# ------------------------------- ingest ---------------------------------
def test_ingest_copies_to_raw(png_file, settings):
    result = ingest_media(png_file, settings)
    assert result.stored_path.exists()
    assert result.stored_path.parent == settings.raw_dir
    assert not list(settings.raw_dir.glob("*.part"))


def test_ingest_is_idempotent(png_file, settings):
    first = ingest_media(png_file, settings)
    second = ingest_media(png_file, settings)
    assert first.stored_path == second.stored_path
    assert len(list(settings.raw_dir.iterdir())) == 1


def test_ingest_rejects_invalid(tmp_path, settings):
    f = tmp_path / "x.png"
    f.write_text("nope")
    with pytest.raises(MediaValidationError) as info:
        ingest_media(f, settings)
    assert info.value.errors
    assert not settings.raw_dir.exists() or not any(settings.raw_dir.iterdir())
