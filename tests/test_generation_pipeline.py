"""Tests for the image generation explorer. No invented model scores."""

import hashlib

import numpy as np
import pytest

from deeptrace.generation.image_pipeline import (
    SYNTHESIS_PATHS,
    GenerationInputError,
    decode_image_bytes,
    run_image_generation_pipeline,
    sample_portrait,
    sample_texture,
)


def test_builtin_samples_are_rgb_canvases():
    for image in (sample_portrait(), sample_texture()):
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8


def test_decode_rejects_non_images():
    with pytest.raises(GenerationInputError):
        decode_image_bytes(b"not an image")


def test_autoencoder_path_reports_real_loss_and_jpeg():
    run = run_image_generation_pipeline(
        sample_portrait(),
        source_id="portrait",
        source_label="Illustrated portrait",
        synthesis_path="toy_autoencoder",
        ae_steps=8,
        seed=3,
    )
    assert run.ae_ran is True
    assert run.ae_losses[0] > run.ae_losses[-1]
    assert run.manifest["synthesis_kind"] == "real"
    assert run.manifest["generator_weights_loaded"] is False
    assert run.manifest["video_generation"] is False
    assert run.manifest["audio_generation"] is False
    assert run.jpeg_bytes[:3] == b"\xff\xd8\xff"
    assert run.sha256 == hashlib.sha256(run.jpeg_bytes).hexdigest()
    assert run.images["final"].shape == (256, 256, 3)
    assert len(run.diffusion_frames) == 5


@pytest.mark.parametrize("path_name", ["upsampling_demo", "diffusion_forward_demo"])
def test_demonstration_paths_are_labeled(path_name):
    run = run_image_generation_pipeline(
        sample_texture(),
        source_id="texture",
        source_label="Geometric texture",
        synthesis_path=path_name,
        ae_steps=2,
    )
    assert run.ae_ran is False
    assert run.ae_losses == []
    assert run.manifest["synthesis_kind"] == "demonstration"
    assert path_name in SYNTHESIS_PATHS
    assert run.images["synthesis"].shape == (256, 256, 3)


def test_unknown_synthesis_path():
    with pytest.raises(GenerationInputError):
        run_image_generation_pipeline(
            sample_portrait(),
            source_id="portrait",
            source_label="Illustrated portrait",
            synthesis_path="stable_diffusion",
        )
