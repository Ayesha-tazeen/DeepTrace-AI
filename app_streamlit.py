"""DeepTrace AI - Multimodal Deepfake Forensic Investigation Dashboard.

Professional dark cybersecurity interface for Image, Video and Audio forensics.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from deeptrace.config import get_settings
from deeptrace.database.db import get_report, init_db, list_reports
from deeptrace.pipeline import ForensicPipeline
from deeptrace.schemas import AnalysisReport, FinalVerdict, MediaType
from deeptrace.ui.generation_explorer import render_generation_explorer

# Page configuration
st.set_page_config(
    page_title="DeepTrace AI | Multimodal Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Cybersecurity CSS Theme - High Contrast
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }

    /* Streamlit native markdown text, headers, and paragraphs */
    p, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        color: #f1f5f9 !important;
    }

    h1, h2, h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Subheaders and section titles */
    h2 {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin-top: 20px;
    }

    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #030712 !important;
        color: #38bdf8 !important;
        border: 1px solid #1e293b !important;
    }

    /* Cyber Card */
    .cyber-card {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.6);
        color: #f1f5f9 !important;
    }

    /* Status Badges */
    .badge-authentic {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #34d399 !important;
        border: 2px solid #059669 !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        letter-spacing: 1px;
        display: inline-block;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }

    .badge-suspicious {
        background: rgba(245, 158, 11, 0.2) !important;
        color: #fbbf24 !important;
        border: 2px solid #d97706 !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        letter-spacing: 1px;
        display: inline-block;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }

    .badge-deepfake {
        background: rgba(239, 68, 68, 0.25) !important;
        color: #f87171 !important;
        border: 2px solid #dc2626 !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        letter-spacing: 1px;
        display: inline-block;
        font-size: 1.1rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    .badge-inconclusive {
        background: rgba(148, 163, 184, 0.2) !important;
        color: #cbd5e1 !important;
        border: 2px solid #475569 !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
        font-size: 1.1rem;
    }

    /* Pipeline Step Tracker */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 15px 0;
    }

    .pipeline-step {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace !important;
        color: #94a3b8 !important;
        font-weight: 600;
    }

    .pipeline-step.active {
        background: rgba(14, 165, 233, 0.25) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        font-weight: 700;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }

    .pipeline-step.done {
        background: rgba(16, 185, 129, 0.25) !important;
        border-color: #34d399 !important;
        color: #34d399 !important;
        font-weight: 700;
    }

    .pipeline-step.fail {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: #f87171 !important;
        color: #f87171 !important;
        font-weight: 700;
    }

    /* Metrics */
    .metric-box {
        background: #111827 !important;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #1f2937 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #030712 !important;
        border-right: 1px solid #1f2937 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1, #075985) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
        border-color: #7dd3fc !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #047857) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid #34d399 !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #047857, #065f46) !important;
        box-shadow: 0 0 15px rgba(52, 211, 153, 0.4) !important;
    }

    /* File uploader box */
    [data-testid="stFileUploadDropzone"] {
        background-color: #111827 !important;
        border: 2px dashed #374151 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38bdf8 !important;
    }
    [data-testid="stFileUploadDropzone"] * {
        color: #cbd5e1 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        border-radius: 6px !important;
        color: #f1f5f9 !important;
    }

    /* Info / Success / Warning / Error alerts */
    div[data-testid="stAlert"] {
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .kind-real, .kind-demo, .kind-explain {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .kind-real {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399 !important;
        border: 1px solid #059669;
    }
    .kind-demo {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24 !important;
        border: 1px solid #d97706;
    }
    .kind-explain {
        background: rgba(56, 189, 248, 0.15);
        color: #7dd3fc !important;
        border: 1px solid #0284c7;
    }
    .cyber-card h4 {
        margin: 8px 0 6px 0;
        color: #e2e8f0 !important;
    }
    .cyber-card p {
        color: #cbd5e1 !important;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize settings and pipeline
settings = get_settings()
settings.ensure_directories()
init_db(settings.db_path)
pipeline = ForensicPipeline(settings=settings)


def render_pipeline_tracker(timeline: List[Any]):
    """Display interactive horizontal pipeline timeline."""
    stages = [
        "UPLOAD", "VALIDATION", "HASHING", "PREPROCESSING",
        "AI_DETECTION", "FORENSIC_ANALYSIS", "EVIDENCE_AGGREGATION", "FINAL_REPORT"
    ]
    done_stages = {item.stage.value: item.status for item in timeline}

    html = '<div class="pipeline-container">'
    for i, s in enumerate(stages):
        status = done_stages.get(s, "pending")
        cls = "pipeline-step"
        if status == "completed":
            cls += " done"
            prefix = "✓ "
        elif status == "failed":
            cls += " fail"
            prefix = "✕ "
        elif status == "running":
            cls += " active"
            prefix = "▶ "
        else:
            prefix = f"{i+1}. "
        html += f'<div class="{cls}">{prefix}{s}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_verdict_badge(verdict: FinalVerdict, confidence: Optional[float]):
    """Render cybersecurity verdict banner."""
    conf_str = f"({confidence:.1%} CONFIDENCE)" if confidence is not None else "(HEURISTIC AUDIT)"
    if verdict == FinalVerdict.AUTHENTIC:
        badge = f'<div class="badge-authentic">🛡️ VERDICT: AUTHENTIC / NATURAL MEDIA {conf_str}</div>'
    elif verdict == FinalVerdict.SUSPICIOUS:
        badge = f'<div class="badge-suspicious">⚠️ VERDICT: SUSPICIOUS ANOMALIES DETECTED {conf_str}</div>'
    elif verdict == FinalVerdict.DEEPFAKE:
        badge = f'<div class="badge-deepfake">🚨 VERDICT: SYNTHETIC / DEEPFAKE DETECTED {conf_str}</div>'
    else:
        badge = f'<div class="badge-inconclusive">❓ VERDICT: INCONCLUSIVE / VERIFICATION REQUIRED {conf_str}</div>'
    st.markdown(badge, unsafe_allow_html=True)


def render_report_details(
    report: AnalysisReport,
    media_bytes: Optional[bytes] = None,
    key_prefix: str = "report",
):
    """Render full forensic investigation report with artifacts, evidence and models."""
    st.markdown("---")
    render_verdict_badge(report.verdict, report.verdict_confidence)

    st.markdown(f"**Forensic Assessment Summary:** {report.verdict_summary}")

    # Pipeline stages
    render_pipeline_tracker(report.pipeline_timeline)

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{report.media_type.value.upper()}</div><div class="metric-label">Media Type</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        val_status = "VALID" if report.validation.is_valid else "INVALID"
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{val_status}</div><div class="metric-label">Integrity Status</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        flagged_count = sum(1 for a in report.forensics.artifacts if a.flagged)
        color = "#EF4444" if flagged_count >= 2 else ("#F59E0B" if flagged_count == 1 else "#10B981")
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:{color}">{flagged_count}</div><div class="metric-label">Flagged Anomalies</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        models_count = len(report.forensics.models)
        st.markdown(
            f'<div class="metric-box"><div class="metric-value">{models_count}</div><div class="metric-label">AI Models Evaluated</div></div>',
            unsafe_allow_html=True,
        )

    # Technical provenance
    with st.expander("🔍 Technical Metadata & SHA-256 Provenance", expanded=True):
        st.code(
            f"File Name:     {report.source_filename}\n"
            f"SHA-256 Hash:  {report.sha256}\n"
            f"MIME Type:     {report.validation.mime_type or 'unknown'}\n"
            f"File Size:     {report.validation.size_bytes or 0} bytes\n"
            f"Dimensions:    {report.validation.width or 'N/A'} x {report.validation.height or 'N/A'} px\n"
            f"Duration:      {report.validation.duration_s or 'N/A'} s\n"
            f"Sample Rate:   {report.validation.sample_rate or 'N/A'} Hz\n"
            f"Report ID:     {report.report_id}",
            language="yaml",
        )

    # Measurable Forensic Indicators
    if report.forensics.artifacts:
        st.subheader("🔬 Measurable Forensic Artifacts")
        table_data = []
        for a in report.forensics.artifacts:
            flag_icon = "🚨 FLAGGED" if a.flagged else "✅ NOMINAL"
            table_data.append(
                {
                    "Artifact Indicator": a.name,
                    "Code": a.indicator,
                    "Measured Score": a.score,
                    "Threshold": a.threshold,
                    "Audit Status": flag_icon,
                    "Forensic Context": a.description,
                }
            )
        st.dataframe(table_data, use_container_width=True)

    # Genuine AI Model Evaluations
    if report.forensics.models:
        st.subheader("🤖 Pretrained AI Models & Deep Representations")
        for m in report.forensics.models:
            with st.container():
                st.markdown(
                    f"**Model:** `{m.model_name}` ({m.model_type}) — **Weights Status:** `{m.weights_status}`"
                )
                pred_label = m.prediction
                st.write(f"- **Prediction:** `{pred_label}`")
                if m.confidence is not None:
                    st.write(f"- **Confidence:** `{m.confidence:.1%}`")
                st.write(f"- **Details:** {m.details}")
                if m.raw_scores:
                    st.caption("Raw Deep Embeddings & Logits:")
                    st.json(m.raw_scores)

    # Evidence Chain
    st.subheader("⛓️ Explainable Evidence Chain")
    for ev in report.evidence_chain:
        if "FLAGGED" in ev or "CRITICAL" in ev or "ALERT" in ev:
            st.error(ev, icon="🚨")
        elif "NORMAL" in ev or "AUTHENTIC" in ev or "CHECK" in ev:
            st.success(ev, icon="🛡️")
        else:
            st.info(ev, icon="ℹ️")

    # Download Report Button
    st.markdown("---")
    report_json = report.model_dump_json(indent=2)
    st.download_button(
        label="📥 Download Forensic Analysis Report (JSON)",
        data=report_json,
        file_name=f"DeepTrace_Report_{report.sha256[:12]}.json",
        mime="application/json",
        use_container_width=True,
        key=f"dl_btn_{key_prefix}_{report.report_id}",
    )


# Sidebar System Telemetry
with st.sidebar:
    st.title("🛡️ DeepTrace AI")
    st.caption("Multimodal Cybersecurity Deepfake Analysis Engine")
    st.markdown("---")

    st.subheader("Engine Status")
    st.write("🟢 **System:** Online")
    st.write(f"👁️ **Vision Model:** `{pipeline.image_detector.vision_model.weights_status}`")
    st.write(f"🎙️ **Audio Model:** `{pipeline.audio_detector.audio_model.weights_status}`")
    st.write(f"🗄️ **Database:** SQLite Active (`deeptrace.db`)")

    st.markdown("---")
    st.caption(
        "Detection covers image, video, and audio. The last tab is an image-only generation "
        "explorer: real preprocessing and a toy autoencoder, with GAN and diffusion steps marked "
        "as demonstrations. No video or audio is generated."
    )


# Main Tabs
tabs = st.tabs([
    "📸 Image Forensics",
    "🎥 Video Forensics",
    "🎙️ Audio Forensics",
    "📜 Forensic History & Audit Log",
    "🧬 Generation Explorer",
])


# -----------------------------------------------------------------------------
# TAB 1: Image Forensics
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("📸 Image Forensics & Deepfake Detection")
    st.write(
        "Upload a JPEG or PNG image to detect synthetic face generation, periodic GAN/diffusion "
        "high-frequency grid patterns, and compression discrepancies (ELA)."
    )

    image_file = st.file_uploader(
        "Select Image File",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if image_file:
        col_preview, col_btn = st.columns([2, 1])
        with col_preview:
            st.image(image_file, caption=f"Uploaded: {image_file.name}", use_container_width=True)
        with col_btn:
            st.write(f"**Size:** {image_file.size / 1024:.1f} KB")
            analyze_img_btn = st.button("🚀 Run Deep Forensic Pipeline", key="btn_img_analyze", use_container_width=True)

        if analyze_img_btn:
            with st.spinner("Executing pipeline: Upload ➔ Validation ➔ Hashing ➔ Preprocessing ➔ AI Detection ➔ Evidence Aggregation..."):
                # Save temp file
                suffix = Path(image_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(image_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    report = pipeline.run_pipeline(tmp_path, original_filename=image_file.name, persist=True)

                    # Show ELA comparison map
                    img_bgr = pipeline.image_detector.preprocessor.load_image(tmp_path)
                    ela_map, _ = pipeline.image_detector.preprocessor.compute_ela(img_bgr)
                    ela_rgb = cv2.cvtColor(ela_map, cv2.COLOR_BGR2RGB)

                    st.markdown("### 🖼️ Forensic Visual Artifact Maps")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(image_file, caption="Original Asset", use_container_width=True)
                    with c2:
                        st.image(ela_rgb, caption="Error Level Analysis (ELA) Residual Map", use_container_width=True)

                    render_report_details(report, key_prefix="image")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()


# -----------------------------------------------------------------------------
# TAB 2: Video Forensics
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("🎥 Video Multi-Frame & Audio-Visual Forensics")
    st.write(
        "Upload an MP4 or MOV video. The pipeline samples frames, measures optical flow temporal "
        "flicker, extracts facial tracks, and isolates any embedded audio stream for acoustic audit."
    )

    video_file = st.file_uploader(
        "Select Video File",
        type=["mp4", "mov"],
        key="video_uploader",
    )

    if video_file:
        st.video(video_file)
        if st.button("🚀 Analyze Video Stream", key="btn_vid_analyze", use_container_width=True):
            with st.spinner("Sampling video frames, tracking faces & evaluating temporal consistency..."):
                suffix = Path(video_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    report = pipeline.run_pipeline(tmp_path, original_filename=video_file.name, persist=True)
                    render_report_details(report, key_prefix="video")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()


# -----------------------------------------------------------------------------
# TAB 3: Audio Forensics
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("🎙️ Audio Synthetic Speech & Voice Clone Detection")
    st.write(
        "Upload a WAV or MP3 audio file. The pipeline checks for neural vocoder energy cutoffs, "
        "spectral flatness, pause dynamic absence, and extracts 2D spectral CNN features."
    )

    audio_file = st.file_uploader(
        "Select Audio File",
        type=["wav", "mp3"],
        key="audio_uploader",
    )

    if audio_file:
        st.audio(audio_file)
        if st.button("🚀 Analyze Acoustic Signal", key="btn_aud_analyze", use_container_width=True):
            with st.spinner("Extracting STFT spectrogram & evaluating neural acoustic features..."):
                suffix = Path(audio_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    report = pipeline.run_pipeline(tmp_path, original_filename=audio_file.name, persist=True)

                    # Compute and display spectrogram plot
                    sig, sr = pipeline.audio_detector.preprocessor.load_audio(tmp_path)
                    spec = pipeline.audio_detector.preprocessor.compute_spectrogram(sig, sr)

                    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0F172A")
                    ax.set_facecolor("#0F172A")
                    im = ax.imshow(
                        spec,
                        aspect="auto",
                        origin="lower",
                        cmap="magma",
                        extent=[0, len(sig) / sr, 0, sr / 2000],
                    )
                    ax.set_title("Short-Time Fourier Transform (STFT) Spectrogram", color="#00F0FF", fontsize=12)
                    ax.set_xlabel("Time (s)", color="#94A3B8")
                    ax.set_ylabel("Frequency (kHz)", color="#94A3B8")
                    ax.tick_params(colors="#94A3B8")
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Magnitude (dB)", color="#94A3B8")
                    cbar.ax.tick_params(colors="#94A3B8")

                    st.pyplot(fig)
                    plt.close(fig)

                    render_report_details(report, key_prefix="audio")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()


# -----------------------------------------------------------------------------
# TAB 4: Forensic Audit Log & SQLite History
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("📜 Forensic Audit Log & SQLite History")
    st.write("Browse and inspect all historical analyses stored securely in the local SQLite database.")

    history = list_reports(limit=100, db_path=settings.db_path)
    if not history:
        st.info("No analyses recorded in database yet. Upload and analyze an asset above to create audit logs.")
    else:
        st.dataframe(history, use_container_width=True)

        selected_id = st.selectbox(
            "Select Report ID to View Full Investigation Record",
            options=[h["report_id"] for h in history],
        )

        if selected_id:
            historical_report = get_report(selected_id, settings.db_path)
            if historical_report:
                render_report_details(historical_report, key_prefix="history")


# -----------------------------------------------------------------------------
# TAB 5: Image generation pipeline explorer
# -----------------------------------------------------------------------------
with tabs[4]:
    render_generation_explorer(pipeline, render_report_details)
