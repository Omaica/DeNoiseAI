"""
DeNoiseAI - FastAPI Backend
Chay: python app/server.py
Sau do mo trinh duyet: http://localhost:8000
"""

import io
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from scripts.inference import load_model, denoise_audio

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
if not CHECKPOINT_PATH:
    if os.path.exists("checkpoints/best_model.pth"):
        CHECKPOINT_PATH = "checkpoints/best_model.pth"
    elif os.path.exists("checkpoints/quick/best_model.pth"):
        CHECKPOINT_PATH = "checkpoints/quick/best_model.pth"
    else:
        CHECKPOINT_PATH = "checkpoints/best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="DeNoiseAI", description="Speech Enhancement API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
# Load model khi khởi động server
# ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"DeNoiseAI Server starting...")
print(f"Device: {DEVICE}")
print(f"Loading model from: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
    MODEL = None
    CONFIG = {}
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 32000
    EPOCH = None
    NUM_PARAMS = 0
else:
    MODEL, CONFIG = load_model(CHECKPOINT_PATH, DEVICE)
    SAMPLE_RATE = CONFIG.get("audio", {}).get("sample_rate", 16000)
    SEGMENT_LENGTH = CONFIG.get("audio", {}).get("n_samples", 32000)
    NUM_PARAMS = MODEL.get_num_params()
    try:
        ckpt_meta = torch.load(CHECKPOINT_PATH, map_location="cpu")
        EPOCH = ckpt_meta.get("epoch", None)
    except Exception:
        EPOCH = None
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    if EPOCH is not None:
        print(f"Epoch: {EPOCH}")
    print(f"Parameters: {NUM_PARAMS:,}")
    print(f"Model loaded successfully!")
print(f"{'='*50}\n")


# ────────────────────────────────────────────────
# Static files (giao diện HTML)
# ────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve giao dien chinh."""
    html_path = Path("app/static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Kiem tra server va model."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "sample_rate": SAMPLE_RATE,
        "checkpoint": str(CHECKPOINT_PATH).replace("\\", "/"),
        "epoch": EPOCH if "EPOCH" in globals() else None,
        "parameters": NUM_PARAMS if "NUM_PARAMS" in globals() else 0,
    }


@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    """
    Nhan file audio nhieu, tra ve file audio da khu nhieu.
    
    - Input: file WAV/MP3/FLAC (bat ky do dai)
    - Output: file WAV da xu ly (16kHz, mono)
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model chua duoc load. Kiem tra checkpoint.")

    # Kiem tra dinh dang
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Dinh dang khong ho tro: {ext}. Su dung: {allowed_extensions}"
        )

    try:
        # Doc file upload vao bo nho
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio voi torchaudio
        waveform, sr = torchaudio.load(audio_buffer)

        # Chuyen ve mono neu stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        # Resample ve sample_rate cua model
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        original_length = waveform.shape[0]

        # Khu nhieu
        enhanced = denoise_audio(
            MODEL, waveform, DEVICE,
            segment_length=SEGMENT_LENGTH,
            overlap=SEGMENT_LENGTH // 4,
        )

        # Trim ve do dai goc
        enhanced = enhanced[:original_length]

        # Ghi ra buffer WAV
        output_buffer = io.BytesIO()
        torchaudio.save(
            output_buffer,
            enhanced.unsqueeze(0),  # (1, T)
            SAMPLE_RATE,
            format="wav",
        )
        output_buffer.seek(0)

        # Tra ve file WAV
        filename = Path(file.filename).stem + "_denoised.wav"
        return StreamingResponse(
            output_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi xu ly audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
