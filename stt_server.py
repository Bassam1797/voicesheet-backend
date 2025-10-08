from fastapi import FastAPI, UploadFile, File
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
precision = os.getenv("WHISPER_PRECISION", "int8_float16")
model = WhisperModel(model_name, device=device, compute_type=precision)

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(contents)
        tmp.flush()
    segments, info = model.transcribe(
        tmp.name,
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
    os.remove(tmp.name)
    return {"text": text}
