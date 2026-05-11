# Backend Inspection Report

Date: 2026-05-11 (UTC)

## Scope

- `stt_server.py`
- `requirements.txt`
- `README.md`

## Architecture Summary

The backend is a minimal FastAPI service with a single endpoint:

- `POST /api/transcribe`

Request flow:

1. Reads uploaded file bytes into memory.
2. Writes bytes to a temporary file.
3. Calls `faster_whisper.WhisperModel.transcribe(...)` with fixed decoding and VAD settings.
4. Concatenates segment text and returns `{ "text": "..." }`.
5. Deletes temporary file in a `finally` block.

## Findings

### 1) CORS is fully open and allows credentials

- Current CORS configuration uses:
  - `allow_origins=["*"]`
  - `allow_credentials=True`
  - `allow_methods=["*"]`
  - `allow_headers=["*"]`

Risk:

- Overly permissive for production and can violate browser CORS expectations when credentials are enabled with wildcard origins.

Recommendation:

- Restrict origins to known frontend domains via environment variable and disable credentials unless strictly required.

### 2) Uploaded file read strategy can spike memory usage

- `await file.read()` loads the entire file into memory.

Risk:

- Large uploads can cause increased memory pressure or denial-of-service conditions.

Recommendation:

- Stream uploads to disk in chunks and enforce maximum file size.

### 3) No file type/content validation

- The API accepts any uploaded content and forwards it to the transcription model.

Risk:

- Invalid files may waste compute, trigger model-level errors, or increase attack surface.

Recommendation:

- Validate content type and/or probe media format before model invocation; reject unsupported formats with 415.

### 4) Model initialized at import time

- `WhisperModel(...)` is created globally at module import.

Impact:

- Startup can be slow; failed model init crashes app early (which may be desirable), but makes health reporting coarse.

Recommendation:

- Keep this behavior for simplicity, but add `/healthz` and `/readyz` endpoints that reflect model readiness.

### 5) Endpoint lacks explicit error handling for transcription failures

- Exceptions from `model.transcribe` are not translated into structured API errors.

Risk:

- Internal errors may surface as generic 500 without actionable detail for clients.

Recommendation:

- Catch model/runtime exceptions and map to controlled HTTP errors with stable response schema.

### 6) Observability is minimal

- No request logging, latency metrics, or error counters are present.

Risk:

- Difficult to debug performance and production incidents.

Recommendation:

- Add structured logs and basic metrics (request duration, request size, failure count).

### 7) Language is hard-coded to English

- `language="en"` is fixed in the transcription call.

Impact:

- Non-English audio transcriptions are constrained.

Recommendation:

- Allow optional language parameter (validated), or enable auto-detection depending on product requirements.

## Positive Notes

- Temporary file cleanup is correctly handled via `finally`.
- Empty upload rejection (`400`) is implemented.
- Key model settings are environment-configurable (`WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_PRECISION`).

## Suggested Next Steps (Priority Order)

1. Lock down CORS for production domains.
2. Add upload size limits and chunked file handling.
3. Add media type validation.
4. Add health/readiness endpoints and basic observability.
5. Add controlled error mapping for model failures.
6. Decide whether multilingual support is required.
