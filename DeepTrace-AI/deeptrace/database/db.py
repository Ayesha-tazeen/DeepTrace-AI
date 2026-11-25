"""Database module for DeepTrace AI analysis history and SQLite storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from deeptrace.config import Settings, get_settings
from deeptrace.logging_utils import get_logger
from deeptrace.schemas import AnalysisReport

logger = get_logger(__name__)


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the SQLite database schema if not already present."""
    settings = get_settings()
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_reports (
                report_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                verdict TEXT NOT NULL,
                verdict_confidence REAL,
                verdict_summary TEXT,
                report_json TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sha256 ON analysis_reports (sha256);
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_reports (created_at DESC);
            """
        )
        conn.commit()
    logger.info("Database initialized at %s", path)


def save_report(report: AnalysisReport, db_path: Optional[Path] = None) -> None:
    """Persist an analysis report into SQLite."""
    settings = get_settings()
    path = db_path or settings.db_path
    init_db(path)

    report_dict = report.model_dump(mode="json")
    report_json = json.dumps(report_dict, indent=2)

    with sqlite3.connect(str(path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_reports (
                report_id, created_at, source_filename, sha256,
                media_type, verdict, verdict_confidence, verdict_summary, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.report_id,
                report.created_at,
                report.source_filename,
                report.sha256,
                report.media_type.value,
                report.verdict.value,
                report.verdict_confidence,
                report.verdict_summary,
                report_json,
            ),
        )
        conn.commit()
    logger.info("Persisted report %s (SHA: %s)", report.report_id, report.sha256[:12])


def get_report(report_id: str, db_path: Optional[Path] = None) -> Optional[AnalysisReport]:
    """Retrieve a report by its UUID."""
    settings = get_settings()
    path = db_path or settings.db_path
    if not path.exists():
        return None

    with sqlite3.connect(str(path)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT report_json FROM analysis_reports WHERE report_id = ?",
            (report_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return AnalysisReport.model_validate(data)


def list_reports(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List recent reports with summary metadata."""
    settings = get_settings()
    path = db_path or settings.db_path
    if not path.exists():
        return []

    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT report_id, created_at, source_filename, sha256,
                   media_type, verdict, verdict_confidence, verdict_summary
            FROM analysis_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
