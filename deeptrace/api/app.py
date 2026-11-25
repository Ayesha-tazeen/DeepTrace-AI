"""FastAPI application for DeepTrace AI multimodal forensics."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from deeptrace.config import Settings, get_settings
from deeptrace.database.db import get_report, init_db, list_reports
from deeptrace.exceptions import DeepTraceError, MediaValidationError
from deeptrace.logging_utils import get_logger
from deeptrace.pipeline import ForensicPipeline
from deeptrace.schemas import AnalysisReport, MediaType

logger = get_logger(__name__)

settings = get_settings()
init_db(settings.db_path)
pipeline = ForensicPipeline(settings=settings)

app = FastAPI(
    title="DeepTrace AI - Multimodal Forensics API",
    description="Cybersecurity Deepfake & Synthetic Media Detection Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root_index() -> str:
    """Friendly root endpoint pointing to interactive API docs and dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepTrace AI - Forensics API</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0B0F19; color: #E2E8F0; padding: 40px; margin: 0; }
            .card { background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 30px; max-width: 650px; margin: 0 auto; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
            h1 { color: #00F0FF; margin-top: 0; }
            a { color: #10B981; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
            .badge { background: #1F2937; color: #00F0FF; padding: 4px 8px; border-radius: 4px; font-family: monospace; }
            ul { line-height: 1.8; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛡️ DeepTrace AI Forensics API</h1>
            <p>The backend forensics API is online and operational.</p>
            <hr style="border-color: #1F2937; margin: 20px 0;">
            <h3>Available Endpoints & Interfaces:</h3>
            <ul>
                <li>📖 <strong>Interactive Swagger API Docs:</strong> <a href="/docs">/docs</a></li>
                <li>📚 <strong>Alternative ReDoc Docs:</strong> <a href="/redoc">/redoc</a></li>
                <li>🟢 <strong>Health & Telemetry Status:</strong> <a href="/api/health">/api/health</a></li>
                <li>📜 <strong>Forensic Audit History:</strong> <a href="/api/history">/api/history</a></li>
            </ul>
            <hr style="border-color: #1F2937; margin: 20px 0;">
            <p>💡 <em>Looking for the visual cybersecurity dashboard?</em><br>
            Run in your terminal: <span class="badge">streamlit run app_streamlit.py</span> (accessible at <strong>http://localhost:8501</strong>)</p>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint providing status of subsystems and models."""
    return {
        "status": "online",
        "service": "DeepTrace AI Forensics Engine",
        "version": "1.0.0",
        "vision_model_status": pipeline.image_detector.vision_model.weights_status,
        "audio_model_status": pipeline.audio_detector.audio_model.weights_status,
        "database": str(settings.db_path),
    }


def _process_upload(upload: UploadFile, expected_type: Optional[MediaType] = None) -> AnalysisReport:
    """Save upload to temp folder and run forensic pipeline."""
    settings.ensure_directories()
    safe_name = Path(upload.filename or "upload.bin").name
    temp_dest = settings.temp_dir / f"api_upload_{safe_name}"

    try:
        with open(temp_dest, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        report = pipeline.run_pipeline(temp_dest, original_filename=safe_name, persist=True)
        if expected_type and report.media_type != expected_type and report.validation.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded media is {report.media_type.value}, but endpoint expects {expected_type.value}",
            )
        return report
    finally:
        if temp_dest.exists():
            try:
                temp_dest.unlink()
            except Exception:
                pass


@app.post("/api/analyze", response_model=AnalysisReport)
def analyze_media(file: UploadFile = File(...)) -> AnalysisReport:
    """Universal endpoint: automatically detects media type (Image, Video, Audio) and executes forensic pipeline."""
    return _process_upload(file)


@app.post("/api/analyze/image", response_model=AnalysisReport)
def analyze_image(file: UploadFile = File(...)) -> AnalysisReport:
    """Analyze image for deepfake facial indicators, frequency grid anomalies, and compression residuals."""
    return _process_upload(file, expected_type=MediaType.IMAGE)


@app.post("/api/analyze/video", response_model=AnalysisReport)
def analyze_video(file: UploadFile = File(...)) -> AnalysisReport:
    """Analyze video for frame flicker, face tracking anomalies, and extracted audio synthetic markers."""
    return _process_upload(file, expected_type=MediaType.VIDEO)


@app.post("/api/analyze/audio", response_model=AnalysisReport)
def analyze_audio(file: UploadFile = File(...)) -> AnalysisReport:
    """Analyze audio for synthetic speech, vocoder cutoffs, and acoustic voice clone indicators."""
    return _process_upload(file, expected_type=MediaType.AUDIO)


@app.get("/api/reports/{report_id}", response_model=AnalysisReport)
def get_analysis_report(report_id: str) -> AnalysisReport:
    """Retrieve an existing forensic analysis report by its ID."""
    report = get_report(report_id, settings.db_path)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@app.get("/api/history")
def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get list of recent analyses from the SQLite audit database."""
    return list_reports(limit=limit, db_path=settings.db_path)
