from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = os.getenv("WHISPER_MODEL", "tiny")
device = os.getenv("WHISPER_DEVICE", "auto")
precision = os.getenv("WHISPER_PRECISION", "int8")
model = WhisperModel(model_name, device=device, compute_type=precision)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    suffix = os.path.splitext(file.filename or "")[1]
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name

        segments, _ = model.transcribe(
            tmp_path,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            language="en",
            temperature=0.0,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = "".join(seg.text for seg in segments).strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": text}
