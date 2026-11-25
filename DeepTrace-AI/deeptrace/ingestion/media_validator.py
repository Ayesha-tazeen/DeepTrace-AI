"""Media validation and safe ingestion.

Validates files that a user explicitly uploaded for forensic analysis:
extension, MIME type (derived from file content), file signature, size,
dimensions/duration and decodability. Invalid files are reported with clear
error messages; nothing is silently ignored and no values are invented.

Public API
----------
validate_media(path, settings=None, allowed_root=None) -> MediaValidationResult
ingest_media(path, settings=None, allowed_root=None)   -> MediaValidationResult
compute_sha256(path) -> str
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

from deeptrace.config import (
    ALLOWED_EXTENSIONS,
    EXPECTED_MIME,
    EXPECTED_SIGNATURE,
    Settings,
    get_settings,
)
from deeptrace.exceptions import MediaValidationError
from deeptrace.ingestion.path_safety import (
    check_filename,
    resolve_input_path,
    sanitize_filename,
)
from deeptrace.logging_utils import get_logger
from deeptrace.schemas import MediaType, MediaValidationResult

logger = get_logger(__name__)

PathLike = Union[str, os.PathLike]

_HEADER_BYTES = 32
_ISOBMFF_LEGACY_ATOMS = (b"moov", b"mdat", b"wide", b"free", b"skip")


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def compute_sha256(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_signature_family(header: bytes) -> Optional[str]:
    """Identify the container family from the first bytes of a file.

    Returns one of ``jpeg``, ``png``, ``wav``, ``mp3``, ``isobmff`` (MP4/MOV)
    or ``None`` when nothing matches.
    """
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:3] == b"ID3":
        return "mp3"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        layer_bits = (header[1] >> 1) & 0x03
        version_bits = (header[1] >> 3) & 0x03
        if layer_bits != 0 and version_bits != 1:  # 0 / 1 are reserved values
            return "mp3"
    if header[4:8] == b"ftyp" or header[4:8] in _ISOBMFF_LEGACY_ATOMS:
        return "isobmff"
    return None


def _png_dimensions(header: bytes) -> Optional[Tuple[int, int]]:
    if len(header) >= 24 and header[12:16] == b"IHDR":
        width, height = struct.unpack(">II", header[16:24])
        return width, height
    return None


def _jpeg_dimensions(handle: BinaryIO) -> Optional[Tuple[int, int]]:
    """Walk JPEG markers until a Start-Of-Frame marker is found."""
    handle.seek(2)  # skip SOI
    while True:
        byte = handle.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":  # padding bytes
            marker = handle.read(1)
        if not marker:
            return None
        code = marker[0]
        if code in (0x01, 0xD8) or 0xD0 <= code <= 0xD7:
            continue  # markers without a length field
        if code == 0xD9:
            return None
        length_bytes = handle.read(2)
        if len(length_bytes) < 2:
            return None
        (length,) = struct.unpack(">H", length_bytes)
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            data = handle.read(5)
            if len(data) < 5:
                return None
            _precision, height, width = struct.unpack(">BHH", data)
            return width, height
        handle.seek(length - 2, os.SEEK_CUR)


def _import_cv2():
    """Import OpenCV lazily so a missing install gives a clear message."""
    try:
        import cv2  # noqa: WPS433

        return cv2
    except ImportError:
        logger.error("OpenCV (cv2) is not installed; cannot decode image/video.")
        return None


# --------------------------------------------------------------------------
# Content probes (each appends to result.errors / fills result fields)
# --------------------------------------------------------------------------
def _probe_image(path: Path, family: str, settings: Settings, result: MediaValidationResult) -> None:
    with open(path, "rb") as handle:
        header = handle.read(_HEADER_BYTES)
        dims = _png_dimensions(header) if family == "png" else _jpeg_dimensions(handle)
    if dims is None:
        result.errors.append("Image header is corrupted: could not read dimensions.")
        return
    width, height = dims
    if width <= 0 or height <= 0:
        result.errors.append("Image has invalid dimensions.")
        return
    if width * height > settings.max_image_pixels:
        result.errors.append(
            f"Image is too large ({width}x{height} px); limit is {settings.max_image_pixels} pixels."
        )
        return
    if min(width, height) < settings.min_image_side:
        result.errors.append(
            f"Image is too small ({width}x{height}); minimum side is {settings.min_image_side} px."
        )
        return

    cv2 = _import_cv2()
    if cv2 is None:
        result.errors.append("OpenCV is not installed, so the image could not be decoded.")
        return
    import numpy as np  # noqa: WPS433

    data = np.fromfile(str(path), dtype=np.uint8)  # works with Unicode paths on Windows
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        result.errors.append("Image is corrupted: OpenCV could not decode it.")
        return
    result.height, result.width = int(image.shape[0]), int(image.shape[1])


def _probe_video(path: Path, settings: Settings, result: MediaValidationResult) -> None:
    cv2 = _import_cv2()
    if cv2 is None:
        result.errors.append("OpenCV is not installed, so the video could not be opened.")
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            result.errors.append("Video is corrupted or uses an unsupported codec: cannot open.")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if fps <= 0 or fps > 240:
            result.errors.append(f"Video reports an invalid frame rate ({fps}).")
            return
        if frames <= 0:
            result.errors.append("Video reports no frames (corrupted or incomplete file).")
            return
        if width <= 0 or height <= 0:
            result.errors.append("Video reports invalid dimensions.")
            return
        if max(width, height) > settings.max_video_side:
            result.errors.append(f"Video resolution {width}x{height} exceeds the limit.")
            return

        ok, _frame = capture.read()
        if not ok:
            result.errors.append("Video is corrupted: the first frame could not be decoded.")
            return

        duration = frames / fps
        result.fps, result.frame_count = fps, frames
        result.width, result.height = width, height
        result.duration_s = round(duration, 3)
        if duration > settings.max_video_duration_s:
            result.errors.append(
                f"Video is {duration:.1f}s long; limit is {settings.max_video_duration_s:.0f}s."
            )
        elif duration > settings.processing_max_seconds:
            result.warnings.append(
                f"Video is {duration:.1f}s long; only the first "
                f"{settings.processing_max_seconds:.0f}s will be analysed."
            )
    finally:
        capture.release()


def _probe_audio(path: Path, settings: Settings, result: MediaValidationResult) -> None:
    duration = sample_rate = channels = None
    failures = []

    try:
        import soundfile as sf  # noqa: WPS433

        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            duration, sample_rate, channels = info.duration, info.samplerate, info.channels
        else:
            failures.append("soundfile: file contains no audio frames")
    except ImportError:
        failures.append("soundfile is not installed")
        logger.error("soundfile is not installed; trying librosa fallback.")
    except Exception as exc:  # libsndfile raises several error types
        failures.append(f"soundfile: {exc}")

    if duration is None:
        try:
            import librosa  # noqa: WPS433

            duration = float(librosa.get_duration(path=str(path)))
            sample_rate = int(librosa.get_samplerate(str(path)))
        except ImportError:
            failures.append("librosa is not installed")
        except Exception as exc:
            failures.append(f"librosa: {exc}")

    if duration is None or duration <= 0:
        result.errors.append("Audio is corrupted or unreadable. Details: " + "; ".join(failures))
        return

    result.duration_s = round(float(duration), 3)
    result.sample_rate = int(sample_rate) if sample_rate else None
    result.channels = int(channels) if channels else None
    if result.duration_s > settings.max_audio_duration_s:
        result.errors.append(
            f"Audio is {result.duration_s:.1f}s long; limit is {settings.max_audio_duration_s:.0f}s."
        )
    elif result.duration_s > settings.processing_max_seconds:
        result.warnings.append(
            f"Audio is {result.duration_s:.1f}s long; only the first "
            f"{settings.processing_max_seconds:.0f}s will be analysed."
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def _finish(result: MediaValidationResult) -> MediaValidationResult:
    result.is_valid = not result.errors
    if result.is_valid:
        logger.info("Validated %s (%s)", result.source_filename, result.media_type.value if result.media_type else "?")
    else:
        logger.warning("Rejected %s: %s", result.source_filename, "; ".join(result.errors))
    return result


def validate_media(
    path: PathLike,
    settings: Optional[Settings] = None,
    allowed_root: Optional[Path] = None,
) -> MediaValidationResult:
    """Validate one file and return a structured result (never raises for bad input)."""
    settings = settings or get_settings()
    raw = os.fspath(path)
    result = MediaValidationResult(source_path=raw, source_filename=Path(raw).name)

    try:
        # 1. Path + filename safety
        resolved, path_errors = resolve_input_path(raw, allowed_root)
        result.errors.extend(path_errors)
        result.errors.extend(check_filename(Path(raw).name, settings.max_filename_length))
        if result.errors or resolved is None:
            return _finish(result)

        # 2. Extension
        ext = resolved.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            result.errors.append(
                f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
            return _finish(result)
        media_type = MediaType(ALLOWED_EXTENSIONS[ext])
        result.extension, result.media_type = ext, media_type

        # 3. Size
        size = resolved.stat().st_size
        result.size_bytes = size
        if size == 0:
            result.errors.append("File is empty.")
            return _finish(result)
        limit = settings.max_size_bytes(media_type.value)
        if size > limit:
            result.errors.append(
                f"File is {size / 1048576:.1f} MB; the {media_type.value} limit is {limit / 1048576:.0f} MB."
            )
            return _finish(result)

        # 4. File signature + MIME (MIME is derived from content, then compared to extension)
        with open(resolved, "rb") as handle:
            header = handle.read(_HEADER_BYTES)
        family = detect_signature_family(header)
        if family is None or family != EXPECTED_SIGNATURE[ext]:
            result.errors.append(
                f"File content does not match the '{ext}' extension "
                f"(detected: {family or 'unknown'}). The file may be corrupted or renamed."
            )
            return _finish(result)
        result.mime_type = EXPECTED_MIME[ext]

        # 5. Hash (provenance) + content probe
        result.sha256 = compute_sha256(resolved)
        if media_type is MediaType.IMAGE:
            _probe_image(resolved, family, settings, result)
        elif media_type is MediaType.VIDEO:
            _probe_video(resolved, settings, result)
        else:
            _probe_audio(resolved, settings, result)
    except Exception as exc:  # never silent: log with traceback and report
        logger.exception("Unexpected failure while validating %s", raw)
        result.errors.append(f"Unexpected validation failure: {type(exc).__name__}: {exc}")

    return _finish(result)


def ingest_media(
    path: PathLike,
    settings: Optional[Settings] = None,
    allowed_root: Optional[Path] = None,
) -> MediaValidationResult:
    """Validate a file, then copy it into ``data/raw`` under a safe name.

    Raises:
        MediaValidationError: if validation fails or the copy is not byte-identical.
    """
    settings = settings or get_settings()
    result = validate_media(path, settings, allowed_root)
    if not result.is_valid:
        raise MediaValidationError(
            f"Validation failed for '{result.source_filename}'.", errors=list(result.errors), result=result
        )

    settings.ensure_directories()
    stem = sanitize_filename(Path(result.source_filename).stem)
    destination = settings.raw_dir / f"{result.sha256[:12]}_{stem}{result.extension}"
    if destination.resolve().parent != settings.raw_dir.resolve():
        raise MediaValidationError("Unsafe destination path computed.", errors=["unsafe destination"])

    if destination.exists() and compute_sha256(destination) == result.sha256:
        logger.info("File already ingested: %s", destination.name)
        result.stored_path = destination
        return result

    temp_copy = destination.with_name(destination.name + ".part")
    try:
        with open(result.source_path, "rb") as src, open(temp_copy, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
        if compute_sha256(temp_copy) != result.sha256:
            raise MediaValidationError(
                "File changed during ingestion (hash mismatch).", errors=["hash mismatch"], result=result
            )
        os.replace(temp_copy, destination)
    finally:
        if temp_copy.exists():
            temp_copy.unlink()  # clean up partial copies

    result.stored_path = destination
    logger.info("Ingested %s -> %s", result.source_filename, destination)
    return result
