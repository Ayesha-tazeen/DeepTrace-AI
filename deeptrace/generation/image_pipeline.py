"""Six-stage image generation explorer.

Real operations (OpenCV, NumPy, a tiny PyTorch autoencoder trained on the
current image) are separated from demonstrations. No pretrained GAN,
autoencoder, or diffusion checkpoint is bundled with DeepTrace, so this
module never presents a classical simulation as model output.

Video and audio generation are intentionally out of scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from deeptrace.preprocessing.image_processor import ImagePreprocessor

CANVAS = 256
AE_SIZE = 64

SYNTHESIS_PATHS: Dict[str, str] = {
    "toy_autoencoder": (
        "Toy convolutional autoencoder reconstruction. A real network is "
        "trained on this single image. The result is a reconstruction, not a face generator."
    ),
    "upsampling_demo": (
        "Classical nearest-neighbour upsampling. Demonstrates the blocky grid "
        "associated with some GAN decoders. No GAN is executed."
    ),
    "diffusion_forward_demo": (
        "Forward diffusion noising schedule. Demonstrates how a clean image is "
        "corrupted. The reverse sampler is not run, because no diffusion weights are loaded."
    ),
}

DETECTION_LINKS: List[Tuple[str, str]] = [
    (
        "Input bytes",
        "Upload validation checks the extension, magic bytes, and decodability, then stores a SHA-256 custody hash.",
    ),
    (
        "Preprocessing",
        "Face detection, resize, and colour conversion use the same image preprocessor as the detection pipeline.",
    ),
    (
        "GAN-style upsampling",
        "Periodic grid energy is what the 2D FFT high-frequency ratio and radial-variance checks are built to measure.",
    ),
    (
        "Heavy smoothing or added noise",
        "Laplacian noise variance in a face region versus the rest of the frame is the spatial-consistency check.",
    ),
    (
        "JPEG polish",
        "Error Level Analysis recompresses the file and scores residual differences left by a second compression pass.",
    ),
    (
        "Face or centre crop",
        "EfficientNet-B0 extracts real features. A calibrated deepfake probability is reported only when fine-tuned weights are on disk.",
    ),
]


class GenerationInputError(ValueError):
    """Raised when explorer input bytes cannot be decoded as an image."""


class ToyConvAutoencoder(nn.Module):
    """Small encode/decode network used only to show a real reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class GenerationRun:
    """Computed explorer state. Images are RGB uint8 arrays."""

    source_id: str
    source_label: str
    synthesis_path: str
    original_shape: Tuple[int, int, int]
    images: Dict[str, np.ndarray]
    diffusion_frames: List[np.ndarray]
    diffusion_timesteps: List[int]
    metrics: Dict[str, float]
    face_boxes: List[Tuple[int, int, int, int]]
    ae_losses: List[float]
    ae_ran: bool
    ae_parameters: int
    notes: Dict[str, str]
    jpeg_bytes: bytes
    sha256: str
    manifest: Dict[str, object] = field(default_factory=dict)


def sample_portrait(size: int = CANVAS) -> np.ndarray:
    """Procedural illustrated portrait. Not a photograph and not a model sample."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / max(size - 1, 1)
        img[y, :, 0] = int(18 + 28 * t)
        img[y, :, 1] = int(28 + 36 * t)
        img[y, :, 2] = int(48 + 40 * (1.0 - t))

    def _ellipse(center, axes, color, thickness=-1):
        cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness, lineType=cv2.LINE_AA)

    _ellipse((size // 2, int(size * 0.58)), (int(size * 0.34), int(size * 0.42)), (46, 58, 86))
    _ellipse((size // 2, int(size * 0.46)), (int(size * 0.22), int(size * 0.28)), (214, 176, 148))
    _ellipse((size // 2, int(size * 0.30)), (int(size * 0.23), int(size * 0.16)), (72, 48, 36))
    _ellipse((int(size * 0.42), int(size * 0.46)), (int(size * 0.035), int(size * 0.045)), (42, 32, 28))
    _ellipse((int(size * 0.58), int(size * 0.46)), (int(size * 0.035), int(size * 0.045)), (42, 32, 28))
    cv2.line(
        img,
        (int(size * 0.44), int(size * 0.60)),
        (int(size * 0.56), int(size * 0.60)),
        (150, 96, 96),
        max(2, size // 80),
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "SAMPLE",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    return img


def sample_texture(size: int = CANVAS) -> np.ndarray:
    """Geometric texture used to make resize and frequency steps easy to see."""
    yy, xx = np.mgrid[0:size, 0:size]
    radial = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    radial = radial / (radial.max() + 1e-6)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = (40 + 180 * (1.0 - radial)).astype(np.uint8)
    img[:, :, 1] = (70 + 80 * radial).astype(np.uint8)
    img[:, :, 2] = (120 + 100 * np.sin(xx / 9.0) * 0.5 + 40).astype(np.uint8)
    img[::16, :, :] = (img[::16, :, :].astype(np.int16) + 50).clip(0, 255).astype(np.uint8)
    img[:, ::16, :] = (img[:, ::16, :].astype(np.int16) + 30).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 6, (248, 250, 252), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        "TEXTURE",
        (12, size - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 23, 42),
        1,
        cv2.LINE_AA,
    )
    return img


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to RGB uint8."""
    if not data:
        raise GenerationInputError("The uploaded file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise GenerationInputError("Could not decode that file as a JPEG or PNG image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit_canvas(rgb: np.ndarray, size: int = CANVAS) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:, :] = (15, 23, 42)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _fft_preview(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = magnitude / (magnitude.max() + 1e-6)
    colored = cv2.applyColorMap((magnitude * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _train_autoencoder(rgb: np.ndarray, steps: int, seed: int) -> Tuple[np.ndarray, List[float], int]:
    torch.manual_seed(seed)
    small = cv2.resize(rgb, (AE_SIZE, AE_SIZE), interpolation=cv2.INTER_AREA)
    target = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    model = ToyConvAutoencoder()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        recon = model(target)
        loss = loss_fn(recon, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    with torch.no_grad():
        out = model(target).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    recon_small = (out * 255.0).round().astype(np.uint8)
    recon = cv2.resize(recon_small, (CANVAS, CANVAS), interpolation=cv2.INTER_CUBIC)
    return recon, losses, _count_parameters(model)


def _forward_diffusion(rgb: np.ndarray, seed: int = 11) -> Tuple[List[np.ndarray], List[int]]:
    """Standard forward noising process. This is not a reverse generative sampler."""
    timesteps = 1000
    betas = np.linspace(1e-4, 0.02, timesteps, dtype=np.float64)
    alpha_bar = np.cumprod(1.0 - betas)
    show_at = [0, 50, 200, 500, 999]
    x0 = rgb.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x0.shape).astype(np.float32)
    frames: List[np.ndarray] = []
    for t in show_at:
        ab = float(alpha_bar[t])
        xt = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise
        frames.append(np.clip(xt * 255.0, 0, 255).round().astype(np.uint8))
    return frames, show_at


def _upsampling_artifact(rgb: np.ndarray) -> np.ndarray:
    """Nearest-neighbour upsample after a hard downsample. Not a GAN."""
    small = cv2.resize(rgb, (48, 48), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST)


def _postprocess(rgb: np.ndarray) -> Tuple[np.ndarray, bytes]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, d=5, sigmaColor=35, sigmaSpace=35)
    blurred = cv2.GaussianBlur(smooth, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(smooth, 1.35, blurred, -0.35, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", sharp, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise RuntimeError("JPEG encoding failed during post-processing.")
    jpeg_bytes = encoded.tobytes()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG round-trip decode failed during post-processing.")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), jpeg_bytes


def _draw_faces(rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    vis = rgb.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (56, 189, 248), 2)
    return vis


def run_image_generation_pipeline(
    source_rgb: np.ndarray,
    source_id: str,
    source_label: str,
    synthesis_path: str,
    ae_steps: int = 24,
    seed: int = 7,
) -> GenerationRun:
    """Execute the six-stage explorer and return images plus an honest manifest."""
    if synthesis_path not in SYNTHESIS_PATHS:
        raise GenerationInputError(f"Unknown synthesis path: {synthesis_path}")
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        raise GenerationInputError("Expected an RGB image.")
    if ae_steps < 1:
        raise GenerationInputError("Autoencoder steps must be at least 1.")

    original_shape = tuple(int(v) for v in source_rgb.shape)
    canvas = _fit_canvas(source_rgb, CANVAS)

    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    boxes = ImagePreprocessor().detect_faces(bgr)
    face_vis = _draw_faces(canvas, boxes)
    spectrum = _fft_preview(canvas)

    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    mean_luma = float(gray.mean())
    std_luma = float(gray.std())

    ae_model = ToyConvAutoencoder()
    ae_parameters = _count_parameters(ae_model)
    del ae_model

    diffusion_frames, diffusion_steps = _forward_diffusion(canvas, seed=seed + 4)
    upsample = _upsampling_artifact(canvas)

    ae_losses: List[float] = []
    ae_ran = synthesis_path == "toy_autoencoder"
    if ae_ran:
        synthesis, ae_losses, ae_parameters = _train_autoencoder(canvas, steps=ae_steps, seed=seed)
        synthesis_note = (
            f"Real PyTorch training for {ae_steps} Adam steps on this single {AE_SIZE}x{AE_SIZE} "
            "image, then cubic upscale for display. This memorizes one picture. It is not a "
            "pretrained deepfake generator, and the loss values below are the actual training losses."
        )
        synthesis_kind = "real"
    elif synthesis_path == "upsampling_demo":
        synthesis = upsample
        synthesis_note = (
            "Demonstration only. The image was area-downsampled to 48x48 and nearest-neighbour "
            "upsampled back to 256x256. That is the resampling pattern behind some GAN checkerboard "
            "artifacts. No generator network ran."
        )
        synthesis_kind = "demonstration"
    else:
        synthesis = diffusion_frames[-1]
        synthesis_note = (
            "Demonstration only. The displayed frame is timestep 999 of the forward noising "
            "process (almost pure noise). No reverse diffusion sampler ran, and no image was generated."
        )
        synthesis_kind = "demonstration"

    final_rgb, jpeg_bytes = _postprocess(synthesis)
    digest = hashlib.sha256(jpeg_bytes).hexdigest()

    residual = cv2.absdiff(canvas, synthesis)
    residual_mean = float(residual.mean())

    images = {
        "input": canvas,
        "faces": face_vis,
        "spectrum": spectrum,
        "upsample_demo": upsample,
        "synthesis": synthesis,
        "residual": residual,
        "final": final_rgb,
    }

    metrics = {
        "mean_luma": round(mean_luma, 3),
        "std_luma": round(std_luma, 3),
        "face_count": float(len(boxes)),
        "ae_parameters": float(ae_parameters),
        "ae_loss_start": round(ae_losses[0], 6) if ae_losses else -1.0,
        "ae_loss_end": round(ae_losses[-1], 6) if ae_losses else -1.0,
        "synthesis_residual_mean": round(residual_mean, 3),
        "jpeg_bytes": float(len(jpeg_bytes)),
    }

    notes = {
        "input": (
            "Working canvas is 256x256 RGB. Built-in pictures are drawn with OpenCV so the "
            "explorer does not depend on external sample files. Uploads are decoded with OpenCV "
            "and letterboxed. None of these inputs are claimed to be model outputs."
        ),
        "preprocess": (
            "Real operations: long-side fit, letterbox, grayscale statistics, Haar/YuNet face "
            "detection through ImagePreprocessor, and a log-magnitude FFT preview."
        ),
        "models": (
            "Architecture explanations are educational. The only network that can run here is "
            f"the toy autoencoder ({ae_parameters:,} parameters). GAN and diffusion weights are "
            "not installed, so those families are not executed."
        ),
        "synthesis": synthesis_note,
        "post": (
            "Real operations on the stage-4 image: bilateral smoothing, unsharp mask, then a "
            "JPEG encode/decode at quality 75. The JPEG file is what detection receives."
        ),
        "provenance": (
            "The SHA-256 digest is computed on the JPEG bytes. The manifest records which "
            "stages were real and which were demonstrations. Sending the file into DeepTrace "
            "runs the existing forensic pipeline and stores a normal history record."
        ),
    }

    manifest: Dict[str, object] = {
        "pipeline": "image_deepfake_generation_explorer",
        "source_id": source_id,
        "source_label": source_label,
        "original_shape_hwc": list(original_shape),
        "canvas": CANVAS,
        "synthesis_path": synthesis_path,
        "synthesis_kind": synthesis_kind,
        "generator_weights_loaded": False,
        "video_generation": False,
        "audio_generation": False,
        "face_boxes_xywh": [list(b) for b in boxes],
        "autoencoder": {
            "architecture": "ToyConvAutoencoder",
            "parameters": ae_parameters,
            "trained_on_this_image": ae_ran,
            "steps": ae_steps if ae_ran else 0,
            "resolution": AE_SIZE,
            "losses": [round(v, 6) for v in ae_losses],
        },
        "diffusion": {
            "executed": "forward_process_only",
            "timesteps_shown": diffusion_steps,
            "reverse_sampler": False,
        },
        "postprocess": {
            "bilateral_filter": True,
            "unsharp_mask": True,
            "jpeg_quality": 75,
        },
        "sha256": digest,
        "jpeg_size_bytes": len(jpeg_bytes),
        "limits": (
            "No production image generator is bundled. Do not describe explorer outputs as "
            "GAN, diffusion, or deepfake samples unless the manifest synthesis_kind and notes say so."
        ),
    }

    return GenerationRun(
        source_id=source_id,
        source_label=source_label,
        synthesis_path=synthesis_path,
        original_shape=original_shape,  # type: ignore[arg-type]
        images=images,
        diffusion_frames=diffusion_frames,
        diffusion_timesteps=diffusion_steps,
        metrics=metrics,
        face_boxes=boxes,
        ae_losses=ae_losses,
        ae_ran=ae_ran,
        ae_parameters=ae_parameters,
        notes=notes,
        jpeg_bytes=jpeg_bytes,
        sha256=digest,
        manifest=manifest,
    )
