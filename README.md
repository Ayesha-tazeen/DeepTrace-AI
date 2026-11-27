# 🛡️ DeepTrace AI: Multimodal Deepfake Forensic Investigation System

A multimodal deepfake forensic investigation and synthetic media detection system built with **FastAPI**, **Streamlit**, **PyTorch**, **OpenCV**, and **SQLite**.

DeepTrace AI analyzes Images, Videos, and Audio files through a deterministic 8-stage forensic pipeline, combining genuine neural network feature extraction with measurable physical and frequency-domain artifact analysis.

---

## 📌 Table of Contents
1. [Pipeline Architecture](#-pipeline-architecture)
2. [Supported Modalities & Capabilities](#-supported-modalities--capabilities)
3. [Genuine AI & Forensic Indicators](#-genuine-ai--forensic-indicators)
4. [Windows 11 & Python 3.11 Setup](#-windows-11--python-311-setup)
5. [How to Run (FastAPI + Streamlit)](#-how-to-run)
6. [Automated Test Suite](#-automated-test-suite)
7. [API Endpoints](#-api-endpoints)
8. [Hosting & Deployment Guide](#-hosting--deployment-guide)

---

## 🔬 Pipeline Architecture

Every uploaded media asset progresses through a strict, transparent 8-stage pipeline:

```
[UPLOAD] ➔ [VALIDATION] ➔ [HASHING] ➔ [PREPROCESSING] ➔ [AI DETECTION] ➔ [FORENSIC ANALYSIS] ➔ [EVIDENCE AGGREGATION] ➔ [FINAL REPORT]
```

1. **UPLOAD:** File received safely, sanitized against path traversal (`..`, control characters, Windows reserved device names like `CON`, `NUL`).
2. **VALIDATION:** Media integrity check using container file signatures (magic bytes), MIME consistency, decodability, duration and dimension boundaries (no silent fallback).
3. **HASHING:** Cryptographic SHA-256 provenance digest computed for immutable chain of custody.
4. **PREPROCESSING:**
   - **Image:** Face detection with OpenCV YuNet DNN / Haar cascade, Error Level Analysis (ELA) recompression difference, 2D FFT Fourier transform.
   - **Video:** Temporal uniform frame sampling (capped by FPS and duration limits), facial track consistency, optical flow inter-frame motion vectors, embedded audio track extraction via FFmpeg.
   - **Audio:** 16kHz mono resampling, short-time Fourier transform (STFT) log-magnitude spectrogram, RMS energy envelope, zero-crossing rate.
5. **AI DETECTION:** Genuine deep learning models:
   - **Vision:** PyTorch `EfficientNet-B0` deep feature representation & embedding analysis.
   - **Audio:** PyTorch `AcousticSpec-CNN` 2D convolutional neural network for spectral feature embeddings.
   - *Status Reporting:* Models report real weights status honestly (`loaded_pretrained_deepfake`, `architecture_ready_feature_mode`, or `weights_unavailable`). Never fabricates random numbers or fake predictions.
6. **FORENSIC ANALYSIS:**
   - **Error Level Analysis (ELA):** Discrepancy score measuring JPEG quantization variances across spliced regions.
   - **2D Fourier FFT Spectrum:** High-frequency radial energy profile to detect periodic checkerboard/grid artifacts typical of generative upsampling (GAN / Diffusion).
   - **Spatial Noise Variance:** Face ROI vs global background Laplacian noise variance ratio (flags boundary smoothing / face-swapping seam blending).
   - **Temporal Optical Flow Jitter:** Measures inter-frame pixel motion variance (flags frame-by-frame deepfake flicker).
   - **Acoustic Vocoder Cutoff:** Detects high-frequency truncation typical of neural vocoders (HiFi-GAN, WaveGlow).
   - **Spectral Flatness & Dynamic Pauses:** Audits unnatural synthetic speech continuity vs natural human breathing.
7. **EVIDENCE AGGREGATION:** Multi-indicator synthesis combining measured artifact flags and neural predictions into an explainable audit trail.
8. **FINAL REPORT:** JSON forensic report generated with unique UUID, persisted to SQLite (`deeptrace.db`), and made available for instant download.

The Streamlit dashboard adds an **Image Deepfake Generation Pipeline Explorer** (last tab). It walks a still image through input, preprocessing, model families, synthesis, post-processing, and provenance. Real steps are OpenCV processing plus a tiny PyTorch autoencoder trained on that one image. GAN upsampling and diffusion noising are labeled demonstrations: this project does not ship a pretrained generator, and it does not generate video or audio. The last stage can send the resulting JPEG through the existing detection pipeline.

---

## 💻 Windows 11 & Python 3.11 Setup

### Prerequisites
- Windows 11 (or Windows 10)
- Python 3.11+ (64-bit)
- PowerShell (Run as Administrator or standard terminal)

### 1. Clone / Open Repository
```powershell
Set-Location "c:\Users\ayesh\OneDrive\Desktop\DeepTrace-AI"
```

### 2. Create Virtual Environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

You can run both the **FastAPI REST API** and the **Streamlit Cybersecurity Dashboard**:

> **Note:** Streamlit and Uvicorn executables are installed inside Python's `Scripts` folder, which may not be on your PATH automatically. Use the `python -m` prefix shown below — it always works regardless of PATH settings.

### Option A: Launch Streamlit Dark Cybersecurity Dashboard (UI)
```powershell
# Always use python -m streamlit (not bare 'streamlit') to avoid PATH issues
& "C:\Users\ayesh\AppData\Local\Python\bin\python.exe" -m streamlit run app_streamlit.py
```
*(If using a venv: `python -m streamlit run app_streamlit.py`)*

Open your browser at: **`http://localhost:8501`**

### Option B: Launch FastAPI Forensics Server (Backend API)
```powershell
& "C:\Users\ayesh\AppData\Local\Python\bin\python.exe" run_api.py
```
Or with Uvicorn (using python -m):
```powershell
& "C:\Users\ayesh\AppData\Local\Python\bin\python.exe" -m uvicorn deeptrace.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation: **`http://127.0.0.1:8000/docs`**

### Optional: Add Python Scripts to PATH permanently (one-time fix)
To make bare `streamlit` and `uvicorn` commands work in future PowerShell sessions:
```powershell
# Add Python Scripts directory to user PATH permanently
$scriptsPath = "C:\Users\ayesh\AppData\Local\Python\pythoncore-3.15-64\Scripts"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$scriptsPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$scriptsPath", "User")
    Write-Host "✅ Scripts directory added to PATH. Restart PowerShell for it to take effect."
} else {
    Write-Host "✅ Scripts directory already in PATH."
}
```

---

## 🧪 Automated Test Suite

Run the full pytest suite (49 automated unit and integration tests across all pipeline stages, validation, SQLite storage, and API endpoints):

```powershell
pytest
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & model status telemetry |
| `POST` | `/api/analyze` | Universal upload endpoint (Auto-detects Image, Video, or Audio) |
| `POST` | `/api/analyze/image` | Image forensics & deepfake detection |
| `POST` | `/api/analyze/video` | Multi-frame video & audio-visual analysis |
| `POST` | `/api/analyze/audio` | Synthetic speech & voice clone detection |
| `GET` | `/api/reports/{id}` | Retrieve historical forensic report by UUID |
| `GET` | `/api/history` | Retrieve recent analyses from SQLite audit log |

---

## 🌐 How to Host DeepTrace AI

Here are the 4 recommended hosting options depending on your needs:

### 1. Docker & Docker Compose (Fastest & Most Reliable)
The repository includes a ready-to-use `Dockerfile` and `docker-compose.yml`.
```bash
docker compose up -d --build
```
- Streamlit UI available on: `http://<server-ip>:8501`
- FastAPI REST API available on: `http://<server-ip>:8000`

### 2. Streamlit Community Cloud (Free 1-Click Dashboard Hosting)
1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, select `app_streamlit.py` as the entrypoint.
4. Click **Deploy**!

### 3. Cloud Virtual Machine (AWS EC2 / DigitalOcean / Hetzner / GCP)
1. Provision an Ubuntu 22.04 LTS instance (2 vCPU, 4GB RAM minimum).
2. Install Python 3.11, git, and libsndfile:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg libsndfile1 git
   ```
3. Clone repository and install requirements.
4. Set up systemd services or run behind **Nginx** reverse proxy with free Let's Encrypt SSL:
   ```nginx
   server {
       server_name deeptrace.yourdomain.com;

       location /api {
           proxy_pass http://127.0.0.1:8000;
       }

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
   }
   ```

### 4. Serverless / Container Platforms (Render / Railway / Fly.io)
Deploy the included `Dockerfile` directly on **Render**, **Railway**, or **Fly.io** for zero-maintenance containerized hosting.
