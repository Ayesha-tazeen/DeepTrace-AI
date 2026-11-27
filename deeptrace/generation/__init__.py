"""Image deepfake generation pipeline explorer.

Educational and forensic tooling only. This package does not ship a
pretrained GAN or diffusion generator, and it does not generate video or audio.
"""

from deeptrace.generation.image_pipeline import (
    SYNTHESIS_PATHS,
    GenerationInputError,
    GenerationRun,
    decode_image_bytes,
    run_image_generation_pipeline,
    sample_portrait,
    sample_texture,
)

__all__ = [
    "SYNTHESIS_PATHS",
    "GenerationInputError",
    "GenerationRun",
    "decode_image_bytes",
    "run_image_generation_pipeline",
    "sample_portrait",
    "sample_texture",
]
