"""Unit tests for SQLite database operations in DeepTrace AI."""

from pathlib import Path

from deeptrace.database.db import get_report, init_db, list_reports, save_report
from deeptrace.schemas import (
    AnalysisReport,
    FinalVerdict,
    MediaForensicAnalysis,
    MediaType,
    MediaValidationResult,
)


def test_database_init_and_crud(tmp_path: Path):
    db_file = tmp_path / "test_deeptrace.db"
    init_db(db_file)
    assert db_file.exists()

    val = MediaValidationResult(
        is_valid=True,
        source_path=str(tmp_path / "sample.jpg"),
        source_filename="sample.jpg",
        media_type=MediaType.IMAGE,
        sha256="abcdef1234567890",
    )
    forensics = MediaForensicAnalysis(
        media_type=MediaType.IMAGE,
        face_detected=False,
        face_count=0,
    )
    report = AnalysisReport(
        report_id="rep-12345",
        created_at="2026-11-26T00:00:00Z",
        source_filename="sample.jpg",
        sha256="abcdef1234567890",
        media_type=MediaType.IMAGE,
        validation=val,
        forensics=forensics,
        verdict=FinalVerdict.AUTHENTIC,
        verdict_confidence=0.92,
        verdict_summary="Authentic image.",
        evidence_chain=["Test normal check."],
    )

    save_report(report, db_path=db_file)

    retrieved = get_report("rep-12345", db_path=db_file)
    assert retrieved is not None
    assert retrieved.report_id == "rep-12345"
    assert retrieved.verdict == FinalVerdict.AUTHENTIC
    assert retrieved.verdict_confidence == 0.92

    history = list_reports(limit=10, db_path=db_file)
    assert len(history) == 1
    assert history[0]["report_id"] == "rep-12345"
    assert history[0]["sha256"] == "abcdef1234567890"
