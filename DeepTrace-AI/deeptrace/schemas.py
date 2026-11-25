"""Pydantic schemas shared across DeepTrace AI modules."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class PipelineStage(str, Enum):
    UPLOAD = "UPLOAD"
    VALIDATION = "VALIDATION"
    HASHING = "HASHING"
    PREPROCESSING = "PREPROCESSING"
    AI_DETECTION = "AI_DETECTION"
    FORENSIC_ANALYSIS = "FORENSIC_ANALYSIS"
    EVIDENCE_AGGREGATION = "EVIDENCE_AGGREGATION"
    FINAL_REPORT = "FINAL_REPORT"


class FinalVerdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    DEEPFAKE = "DEEPFAKE / AI-GENERATED"
    INCONCLUSIVE = "INCONCLUSIVE"


class MediaValidationResult(BaseModel):
    """Outcome of validating (and optionally ingesting) one uploaded file."""

    is_valid: bool = False
    source_path: str
    source_filename: str = ""
    stored_path: Optional[Path] = None

    media_type: Optional[MediaType] = None
    extension: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None

    # Content properties (filled only when actually measured - never guessed)
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PipelineProgress(BaseModel):
    stage: PipelineStage
    status: str = "completed"  # pending, running, completed, failed, skipped
    detail: str = ""
    timestamp: str = ""


class ForensicArtifact(BaseModel):
    """Specific measurable anomaly or forensic indicator."""

    name: str
    indicator: str
    score: float = 0.0
    threshold: float = 0.5
    flagged: bool = False
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelPrediction(BaseModel):
    """AI Model output details with honest reporting of weights/status."""

    model_name: str
    model_type: str
    weights_status: str  # e.g. "loaded_pretrained", "feature_extractor_evaluating", "weights_unavailable"
    prediction: str      # e.g. "AUTHENTIC", "DEEPFAKE / AI-GENERATED", "UNCERTAIN"
    confidence: Optional[float] = None  # None if weights not loaded; never fabricated
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    details: str = ""


class MediaForensicAnalysis(BaseModel):
    """Detailed forensic findings from preprocessing, models, and analytical checks."""

    media_type: MediaType
    face_detected: bool = False
    face_count: int = 0
    face_bounding_boxes: List[List[int]] = Field(default_factory=list)
    frames_analyzed: Optional[int] = None

    artifacts: List[ForensicArtifact] = Field(default_factory=list)
    models: List[ModelPrediction] = Field(default_factory=list)

    frequency_analysis: Dict[str, Any] = Field(default_factory=dict)
    compression_analysis: Dict[str, Any] = Field(default_factory=dict)
    spatial_consistency: Dict[str, Any] = Field(default_factory=dict)
    temporal_consistency: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    audio_included: bool = False


class AnalysisReport(BaseModel):
    """Complete forensic assessment report."""

    report_id: str
    created_at: str
    source_filename: str
    sha256: str
    media_type: MediaType

    validation: MediaValidationResult
    forensics: MediaForensicAnalysis

    verdict: FinalVerdict
    verdict_confidence: Optional[float] = None
    verdict_summary: str
    evidence_chain: List[str] = Field(default_factory=list)
    pipeline_timeline: List[PipelineProgress] = Field(default_factory=list)

