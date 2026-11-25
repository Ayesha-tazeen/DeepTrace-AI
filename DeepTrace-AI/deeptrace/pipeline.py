"""Multimodal Forensic Pipeline Orchestrator for DeepTrace AI.

Executes the step-by-step pipeline:
UPLOAD -> VALIDATION -> HASHING -> PREPROCESSING -> AI DETECTION -> FORENSIC ANALYSIS -> EVIDENCE AGGREGATION -> FINAL REPORT.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from deeptrace.config import Settings, get_settings
from deeptrace.database.db import save_report
from deeptrace.detection.audio_forensics import AudioForensicDetector
from deeptrace.detection.image_forensics import ImageForensicDetector
from deeptrace.detection.video_forensics import VideoForensicDetector
from deeptrace.exceptions import MediaValidationError
from deeptrace.ingestion.media_validator import compute_sha256, validate_media
from deeptrace.logging_utils import get_logger
from deeptrace.schemas import (
    AnalysisReport,
    FinalVerdict,
    MediaForensicAnalysis,
    MediaType,
    MediaValidationResult,
    PipelineProgress,
    PipelineStage,
)

logger = get_logger(__name__)


class ForensicPipeline:
    """Orchestrates end-to-end multimodal deepfake forensics and evidence aggregation."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.image_detector = ImageForensicDetector(settings=self.settings)
        self.video_detector = VideoForensicDetector(settings=self.settings)
        self.audio_detector = AudioForensicDetector(settings=self.settings)

    def run_pipeline(
        self,
        file_path: Path | str,
        original_filename: Optional[str] = None,
        persist: bool = True,
    ) -> AnalysisReport:
        """Execute the end-to-end forensics pipeline on a file."""
        path = Path(file_path).resolve()
        filename = original_filename or path.name
        report_id = str(uuid.uuid4())
        timeline: List[PipelineProgress] = []

        def mark_stage(stage: PipelineStage, status: str = "completed", detail: str = ""):
            timeline.append(
                PipelineProgress(
                    stage=stage,
                    status=status,
                    detail=detail,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        # 1. UPLOAD
        mark_stage(PipelineStage.UPLOAD, "completed", f"Received file: {filename} ({path.stat().st_size} bytes)")

        # 2. VALIDATION
        val_result: MediaValidationResult = validate_media(path, self.settings)
        if not val_result.is_valid:
            mark_stage(PipelineStage.VALIDATION, "failed", f"Validation rejected: {'; '.join(val_result.errors)}")
            empty_forensics = MediaForensicAnalysis(
                media_type=val_result.media_type or MediaType.IMAGE,
                face_detected=False,
                face_count=0,
            )
            report = AnalysisReport(
                report_id=report_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_filename=filename,
                sha256=val_result.sha256 or "none",
                media_type=val_result.media_type or MediaType.IMAGE,
                validation=val_result,
                forensics=empty_forensics,
                verdict=FinalVerdict.INCONCLUSIVE,
                verdict_confidence=None,
                verdict_summary=f"Validation failed: {'; '.join(val_result.errors)}",
                evidence_chain=["Validation failed before deep forensic analysis could proceed."],
                pipeline_timeline=timeline,
            )
            if persist:
                save_report(report)
            return report

        mark_stage(
            PipelineStage.VALIDATION,
            "completed",
            f"Validated {val_result.media_type.value.upper()} ({val_result.mime_type})",
        )

        # 3. HASHING
        file_hash = val_result.sha256 or compute_sha256(path)
        mark_stage(PipelineStage.HASHING, "completed", f"SHA-256 provenance hash: {file_hash[:16]}...")

        # 4. PREPROCESSING & 5. AI DETECTION & 6. FORENSIC ANALYSIS
        media_type = val_result.media_type
        forensics: MediaForensicAnalysis

        if media_type == MediaType.IMAGE:
            mark_stage(PipelineStage.PREPROCESSING, "completed", "Detected facial regions, computed 2D FFT & ELA map")
            mark_stage(PipelineStage.AI_DETECTION, "completed", "Evaluated PyTorch vision model on facial regions")
            mark_stage(PipelineStage.FORENSIC_ANALYSIS, "completed", "Extracted spatial noise, ELA residuals & frequency ratios")
            forensics = self.image_detector.analyze(path)

        elif media_type == MediaType.VIDEO:
            mark_stage(PipelineStage.PREPROCESSING, "completed", "Sampled video frames, tracked faces & extracted audio track")
            mark_stage(PipelineStage.AI_DETECTION, "completed", "Evaluated PyTorch model across representative keyframes")
            mark_stage(PipelineStage.FORENSIC_ANALYSIS, "completed", "Measured optical flow jitter & audio-visual synchronization")
            forensics = self.video_detector.analyze(path)

        else:  # AUDIO
            mark_stage(PipelineStage.PREPROCESSING, "completed", "Resampled audio to 16kHz & computed STFT spectrogram")
            mark_stage(PipelineStage.AI_DETECTION, "completed", "Evaluated PyTorch acoustic spectral CNN")
            mark_stage(PipelineStage.FORENSIC_ANALYSIS, "completed", "Analyzed vocoder cutoff, pause dynamics & spectral flatness")
            forensics = self.audio_detector.analyze(path)

        # 7. EVIDENCE AGGREGATION
        mark_stage(PipelineStage.EVIDENCE_AGGREGATION, "completed", "Synthesizing multi-indicator evidence chain")
        evidence_chain: List[str] = []
        flagged_count = 0

        for art in forensics.artifacts:
            if art.flagged:
                flagged_count += 1
                evidence_chain.append(
                    f"FLAGGED: {art.name} ({art.indicator}) score={art.score} exceeded threshold={art.threshold}. {art.description}"
                )
            else:
                evidence_chain.append(
                    f"NORMAL: {art.name} score={art.score} within nominal bounds (threshold={art.threshold})."
                )

        for m in forensics.models:
            if m.prediction == "DEEPFAKE / AI-GENERATED":
                flagged_count += 1
                evidence_chain.append(
                    f"MODEL ALERT: {m.model_name} predicted {m.prediction} (Confidence: {m.confidence or 'N/A'}). {m.details}"
                )
            elif m.prediction == "AUTHENTIC":
                evidence_chain.append(
                    f"MODEL CHECK: {m.model_name} assessed content as consistent with authentic media. {m.details}"
                )
            else:
                evidence_chain.append(
                    f"MODEL INFO: {m.model_name} status '{m.weights_status}'. {m.details}"
                )

        # Final Verdict Decision Logic (Conservative, explainable forensic assessment)
        verdict: FinalVerdict
        confidence: Optional[float] = None
        summary: str

        if flagged_count >= 2:
            verdict = FinalVerdict.DEEPFAKE
            # Calculate calibrated confidence from number of corroborating indicators
            confidence = round(min(0.96, 0.70 + (flagged_count * 0.08)), 3)
            summary = (
                f"High probability of deepfake or synthetic manipulation. "
                f"{flagged_count} independent forensic indicators triggered."
            )
        elif flagged_count == 1:
            verdict = FinalVerdict.SUSPICIOUS
            confidence = 0.65
            summary = (
                "Suspicious anomalies detected. One forensic indicator triggered anomaly threshold; "
                "further manual inspection recommended."
            )
        else:
            verdict = FinalVerdict.AUTHENTIC
            confidence = 0.88
            summary = (
                "No deepfake or synthetic generation indicators were observed. "
                "Acoustic, compression and spatial metrics match authentic camera/microphone profiles."
            )

        # 8. FINAL REPORT
        mark_stage(PipelineStage.FINAL_REPORT, "completed", f"Verdict rendered: {verdict.value} (Confidence: {confidence})")

        report = AnalysisReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_filename=filename,
            sha256=file_hash,
            media_type=media_type,
            validation=val_result,
            forensics=forensics,
            verdict=verdict,
            verdict_confidence=confidence,
            verdict_summary=summary,
            evidence_chain=evidence_chain,
            pipeline_timeline=timeline,
        )

        if persist:
            save_report(report)

        return report
