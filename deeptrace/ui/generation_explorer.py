"""Streamlit view for the image generation pipeline explorer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import streamlit as st

from deeptrace.generation.image_pipeline import (
    DETECTION_LINKS,
    SYNTHESIS_PATHS,
    GenerationInputError,
    GenerationRun,
    decode_image_bytes,
    run_image_generation_pipeline,
    sample_portrait,
    sample_texture,
)

STAGE_LABELS = [
    "1 · Input",
    "2 · Preprocess",
    "3 · Models",
    "4 · Synthesis",
    "5 · Post-process",
    "6 · Provenance",
]

_KIND_CLASS = {
    "real": "kind-real",
    "demonstration": "kind-demo",
    "explanation": "kind-explain",
}


def _badge(kind: str, label: str) -> None:
    css = _KIND_CLASS.get(kind, "kind-explain")
    st.markdown(f'<span class="{css}">{label}</span>', unsafe_allow_html=True)


def _flow(active: int) -> None:
    names = [
        "Input",
        "Preprocess",
        "Models",
        "Synthesis",
        "Post-process",
        "Provenance",
    ]
    html = ['<div class="pipeline-container">']
    for index, name in enumerate(names):
        cls = "pipeline-step done" if index < active else "pipeline-step"
        if index == active:
            cls = "pipeline-step active"
        html.append(f'<div class="{cls}">{index + 1}. {name}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _examine(pipeline, jpeg_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as handle:
        handle.write(jpeg_bytes)
        path = Path(handle.name)
    try:
        return pipeline.run_pipeline(
            path,
            original_filename="explorer_output.jpg",
            persist=True,
        )
    finally:
        if path.exists():
            path.unlink()


def _loss_figure(losses: list[float]):
    fig, ax = plt.subplots(figsize=(6.2, 2.6), facecolor="#0B0F19")
    ax.set_facecolor("#111827")
    ax.plot(range(1, len(losses) + 1), losses, color="#38bdf8", linewidth=2)
    ax.set_title("Toy autoencoder MSE (measured)", color="#e2e8f0", fontsize=11)
    ax.set_xlabel("Step", color="#94a3b8")
    ax.set_ylabel("MSE", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    fig.tight_layout()
    return fig


def _model_cards(run: GenerationRun | None) -> None:
    param_text = f"{run.ae_parameters:,} parameters" if run else "instantiated when the explorer runs"
    trained = "trained on this image" if run and run.ae_ran else "not trained for the current path"
    cards = [
        (
            "explanation",
            "EXPLANATION",
            "GANs",
            "A generator maps noise to an image while a discriminator learns to reject fakes. "
            "Many decoders upsample with transposed convolutions, which can leave a periodic grid. "
            "No GAN weights are loaded, so DeepTrace does not invent a GAN sample.",
        ),
        (
            "real" if run and run.ae_ran else "explanation",
            "REAL NETWORK" if run and run.ae_ran else "ARCHITECTURE ONLY",
            "Autoencoders",
            f"An encoder compresses the image and a decoder rebuilds it. The toy network here "
            f"({param_text}, {trained}) is a real PyTorch module. It reconstructs one canvas. "
            "It is not a face-swap or text-to-image model.",
        ),
        (
            "demonstration",
            "FORWARD PROCESS ONLY",
            "Diffusion",
            "Diffusion trainers corrupt an image over hundreds of steps, then learn to reverse "
            "that noise. This explorer draws the forward schedule with NumPy. The reverse "
            "sampler stays off because no diffusion checkpoint is installed.",
        ),
    ]
    cols = st.columns(3)
    for col, (kind, badge, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="cyber-card"><span class="{_KIND_CLASS[kind]}">{badge}</span>'
                f"<h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )


def _render_input(run: GenerationRun | None, preview) -> None:
    _badge("real", "REAL · decode and canvas fit")
    st.markdown(run.notes["input"] if run else (
        "Choose a built-in drawing or upload a JPEG/PNG, then run the explorer. "
        "Built-in images are procedural drawings, not photographs and not generator outputs."
    ))
    left, right = st.columns(2)
    with left:
        st.image(preview, caption="Selected input", use_container_width=True)
    with right:
        if run:
            st.image(run.images["input"], caption="256×256 working canvas", use_container_width=True)
            st.write(
                f"Original shape HWC: `{run.original_shape}` · "
                f"Source: **{run.source_label}**"
            )
        else:
            st.info("Run the explorer to letterbox this input onto the 256×256 working canvas.")


def _render_preprocess(run: GenerationRun | None) -> None:
    _badge("real", "REAL · preprocessor, faces, FFT preview")
    if run is None:
        st.info("This stage resizes the input, looks for faces, and draws a log-magnitude spectrum.")
        return
    st.markdown(run.notes["preprocess"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(run.images["input"], caption="Canvas", use_container_width=True)
    with c2:
        caption = f"Face boxes: {len(run.face_boxes)}"
        st.image(run.images["faces"], caption=caption, use_container_width=True)
    with c3:
        st.image(run.images["spectrum"], caption="Log-magnitude FFT (display only)", use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean luma", f"{run.metrics['mean_luma']:.1f}")
    m2.metric("Luma std", f"{run.metrics['std_luma']:.1f}")
    m3.metric("Faces found", int(run.metrics["face_count"]))
    st.caption(
        "A face count of zero on the illustrated sample is an honest detector result. "
        "The same ImagePreprocessor is used by image forensics."
    )


def _render_models(run: GenerationRun | None) -> None:
    _badge("explanation", "EXPLANATION · one real toy network, no generative checkpoint")
    if run:
        st.markdown(run.notes["models"])
    else:
        st.markdown(
            "GAN, autoencoder, and diffusion families are explained side by side. "
            "Only the toy autoencoder can actually run, and only when that synthesis path is selected."
        )
    _model_cards(run)
    if run:
        st.markdown("**Forward diffusion frames (demonstration, shared noise)**")
        cols = st.columns(len(run.diffusion_frames))
        for col, frame, step in zip(cols, run.diffusion_frames, run.diffusion_timesteps):
            with col:
                st.image(frame, caption=f"t={step}", use_container_width=True)
        st.caption("Same noise sample at five timesteps. t=0 is the canvas. Later frames are not generated images.")


def _render_synthesis(run: GenerationRun | None) -> None:
    if run is None:
        _badge("explanation", "WAITING FOR A RUN")
        st.info("Synthesis shows either the toy reconstruction or a clearly labeled classical demonstration.")
        return
    kind = "real" if run.ae_ran else "demonstration"
    label = "REAL · single-image reconstruction" if run.ae_ran else "DEMONSTRATION · not a generative model"
    _badge(kind, label)
    st.markdown(run.notes["synthesis"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(run.images["input"], caption="Canvas", use_container_width=True)
    with c2:
        st.image(run.images["synthesis"], caption="Stage output", use_container_width=True)
    with c3:
        st.image(run.images["residual"], caption="Absolute residual vs canvas", use_container_width=True)
    if run.ae_ran and run.ae_losses:
        fig = _loss_figure(run.ae_losses)
        st.pyplot(fig)
        plt.close(fig)
        st.write(
            f"Measured MSE {run.metrics['ae_loss_start']:.5f} → {run.metrics['ae_loss_end']:.5f} "
            f"over {len(run.ae_losses)} steps · {run.ae_parameters:,} parameters."
        )
    if not run.ae_ran:
        st.image(
            run.images["upsample_demo"],
            caption="Nearest-neighbour upsampling reference (always computed, not a GAN)",
            use_container_width=True,
        )


def _render_post(run: GenerationRun | None) -> None:
    _badge("real", "REAL · smooth, sharpen, JPEG quality 75")
    if run is None:
        st.info("Post-processing is applied to whatever stage 4 produced, then the file is JPEG-encoded.")
        return
    st.markdown(run.notes["post"])
    c1, c2 = st.columns(2)
    with c1:
        st.image(run.images["synthesis"], caption="Before post-processing", use_container_width=True)
    with c2:
        st.image(run.images["final"], caption="After JPEG round-trip", use_container_width=True)
    st.metric("JPEG size", f"{int(run.metrics['jpeg_bytes'])} bytes")


def _render_provenance(run: GenerationRun | None, pipeline, render_report: Callable) -> None:
    _badge("real", "REAL · SHA-256, manifest, optional DeepTrace scan")
    if run is None:
        st.info("The last stage hashes the JPEG and can hand that exact file to the existing detection pipeline.")
        return
    st.markdown(run.notes["provenance"])
    st.image(run.images["final"], caption="Final explorer output", use_container_width=True)
    st.code(f"SHA-256  {run.sha256}\nBytes    {len(run.jpeg_bytes)}\nPath     {run.synthesis_path}", language="text")
    st.json(run.manifest)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download final JPEG",
            data=run.jpeg_bytes,
            file_name="deeptrace_explorer_output.jpg",
            mime="image/jpeg",
            use_container_width=True,
            key="explorer_dl_jpeg",
        )
    with d2:
        st.download_button(
            "Download provenance manifest",
            data=json.dumps(run.manifest, indent=2),
            file_name="deeptrace_explorer_manifest.json",
            mime="application/json",
            use_container_width=True,
            key="explorer_dl_manifest",
        )

    st.markdown("#### How this file meets the detector")
    for title, detail in DETECTION_LINKS:
        st.markdown(f"**{title}.** {detail}")
    st.caption(
        "A detection report on this file describes this JPEG only. It is not a claim that the "
        "same scores would hold for a production generator."
    )
    if st.button("Examine this JPEG with DeepTrace detection", key="explorer_detect", use_container_width=True):
        with st.spinner("Running the existing image forensic pipeline on the explorer JPEG..."):
            report = _examine(pipeline, run.jpeg_bytes)
        render_report(report, key_prefix="explorer")


def render_generation_explorer(pipeline, render_report: Callable) -> None:
    """Interactive six-stage image generation explorer."""
    st.header("Image Deepfake Generation Pipeline Explorer")
    st.markdown(
        '<div class="cyber-card">This page shows how a still image moves from input to a '
        "finished file, and how DeepTrace would examine that file. "
        "<b>Green tags are operations that actually run. Amber tags are demonstrations. "
        "Blue tags are explanations.</b> No video or audio is generated. "
        "No pretrained GAN or diffusion model is loaded, so those outputs are never invented."
        "</div>",
        unsafe_allow_html=True,
    )

    sources: Dict[str, object] = {
        "Illustrated portrait": sample_portrait(),
        "Geometric texture": sample_texture(),
    }
    left, right = st.columns([1.2, 1])
    with left:
        source_name = st.radio(
            "Input source",
            list(sources.keys()) + ["Upload an image"],
            horizontal=True,
            key="explorer_source",
        )
        upload = None
        if source_name == "Upload an image":
            upload = st.file_uploader(
                "JPEG or PNG used only inside this explorer",
                type=["jpg", "jpeg", "png"],
                key="explorer_upload",
            )
    with right:
        path_labels = {
            "toy_autoencoder": "Toy autoencoder (real reconstruction)",
            "upsampling_demo": "Upsampling grid (demonstration)",
            "diffusion_forward_demo": "Diffusion noise (demonstration)",
        }
        synthesis_path = st.radio(
            "What stage 4 should produce",
            list(path_labels.keys()),
            format_func=lambda key: path_labels[key],
            key="explorer_path",
        )
        st.caption(SYNTHESIS_PATHS[synthesis_path])

    preview = sources.get(source_name)
    source_id = source_name
    ready = source_name != "Upload an image"
    if upload is not None:
        try:
            preview = decode_image_bytes(upload.getvalue())
            source_id = f"upload:{upload.name}:{upload.size}"
            ready = True
        except GenerationInputError as exc:
            st.error(str(exc))
            preview = None
            ready = False
    elif source_name == "Upload an image":
        st.info("Upload a JPEG or PNG to use your own still image. Video and audio files are not accepted here.")

    run_clicked = st.button(
        "Run image pipeline explorer",
        type="primary",
        use_container_width=True,
        disabled=not ready,
        key="explorer_run",
    )
    if run_clicked and preview is not None:
        with st.spinner("Fitting the canvas, running the selected stage-4 path, and writing the JPEG..."):
            try:
                st.session_state["gen_explorer_run"] = run_image_generation_pipeline(
                    preview,
                    source_id=source_id,
                    source_label=upload.name if upload is not None else source_name,
                    synthesis_path=synthesis_path,
                )
                st.session_state["gen_explorer_stage"] = 0
            except GenerationInputError as exc:
                st.error(str(exc))

    run: GenerationRun | None = st.session_state.get("gen_explorer_run")
    if run is not None and (run.source_id != source_id or run.synthesis_path != synthesis_path):
        st.warning("The input or synthesis path changed. Run the explorer again to refresh the six stages.")

    if "gen_explorer_stage" not in st.session_state:
        st.session_state["gen_explorer_stage"] = 0
    nav = st.columns(6)
    for index, label in enumerate(STAGE_LABELS):
        if nav[index].button(label, key=f"explorer_stage_{index}", use_container_width=True):
            st.session_state["gen_explorer_stage"] = index
    stage = int(st.session_state["gen_explorer_stage"])
    _flow(stage)

    st.markdown("---")
    if stage == 0:
        _render_input(run, preview if preview is not None else sample_portrait())
    elif stage == 1:
        _render_preprocess(run)
    elif stage == 2:
        _render_models(run)
    elif stage == 3:
        _render_synthesis(run)
    elif stage == 4:
        _render_post(run)
    else:
        _render_provenance(run, pipeline, render_report)
