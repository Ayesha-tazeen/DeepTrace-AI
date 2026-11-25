"""End-to-end tests for the 8-stage forensic pipeline."""

import json

from deeptrace.pipeline import ForensicPipeline
from deeptrace.schemas import FinalVerdict, MediaType, PipelineStage


def test_pipeline_on_image(png_file, settings, tmp_path):
    settings.db_path = tmp_path / "test_pipe.db"
    pipeline = ForensicPipeline(settings)

    report = pipeline.run_pipeline(png_file, original_filename="test_image.png")
    assert report.media_type is MediaType.IMAGE
    assert report.validation.is_valid
    assert len(report.sha256) == 64
    assert report.verdict in [FinalVerdict.AUTHENTIC, FinalVerdict.SUSPICIOUS, FinalVerdict.DEEPFAKE]

    # Verify all 8 stages were recorded in timeline
    recorded_stages = [p.stage for p in report.pipeline_timeline]
    expected_stages = [
        PipelineStage.UPLOAD,
        PipelineStage.VALIDATION,
        PipelineStage.HASHING,
        PipelineStage.PREPROCESSING,
        PipelineStage.AI_DETECTION,
        PipelineStage.FORENSIC_ANALYSIS,
        PipelineStage.EVIDENCE_AGGREGATION,
        PipelineStage.FINAL_REPORT,
    ]
    for stage in expected_stages:
        assert stage in recorded_stages

    # Verify JSON serialization works without error
    json_str = report.model_dump_json()
    data = json.loads(json_str)
    assert data["report_id"] == report.report_id


def test_pipeline_on_audio(wav_file, settings, tmp_path):
    settings.db_path = tmp_path / "test_pipe_audio.db"
    pipeline = ForensicPipeline(settings)

    report = pipeline.run_pipeline(wav_file, original_filename="test_audio.wav")
    assert report.media_type is MediaType.AUDIO
    assert report.validation.is_valid
    assert report.verdict in [FinalVerdict.AUTHENTIC, FinalVerdict.SUSPICIOUS, FinalVerdict.DEEPFAKE]


def test_pipeline_on_video(mp4_file, settings, tmp_path):
    settings.db_path = tmp_path / "test_pipe_video.db"
    pipeline = ForensicPipeline(settings)

    report = pipeline.run_pipeline(mp4_file, original_filename="test_video.mp4")
    assert report.media_type is MediaType.VIDEO
    assert report.validation.is_valid


def test_pipeline_on_invalid_file(tmp_path, settings):
    settings.db_path = tmp_path / "test_pipe_invalid.db"
    fake = tmp_path / "fake.png"
    fake.write_text("not an image")

    pipeline = ForensicPipeline(settings)
    report = pipeline.run_pipeline(fake, original_filename="fake.png")
    assert not report.validation.is_valid
    assert report.verdict is FinalVerdict.INCONCLUSIVE
