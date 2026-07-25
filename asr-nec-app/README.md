# ASR NEC App

Full-stack demonstration application for the ASR named entity correction
pipeline: Whisper transcription, SpeechSearcher candidate retrieval and
GenerativeLabeler error-span correction.

## Local development

Backend tests do not require a GPU because inference is replaced by a fake
engine:

```bash
uv sync --extra dev
uv run pytest
```

Start the real backend (requires workspace model artifacts):

```bash
uv sync --extra nec
uv run python main.py
```

Start the frontend development server:

```bash
cd ui
npm install
npm run dev
```

## GPU deployment

From `asr-nec-app/docker`:

```bash
docker compose build
docker compose up -d
docker compose logs -f app
```

The compose service mounts workspace-level `data`, `weights` and `runs`
read-only. Demo history and uploaded audio live in the persistent `storage`
volume. The application listens on port `8016` by default.
Compose joins the existing external `ai-base` network so Nginx Proxy Manager
can reach the service at `http://asr-nec-demo:8016` without allocating another
Docker subnet.

Production transcription uses the existing `audio_server` service on
`ai-base`. The uploaded audio is normalized to 16 kHz mono WAV for
`POST /audio/transcription`; its text is passed into SS + GL while the same
original audio is encoded locally. Set `AUDIO_API_ASR_ENABLED=false` to use
local Whisper transcription only. When the external service fails,
`AUDIO_API_FALLBACK_TO_WHISPER=true` keeps the demo available.

## API

- `GET /api/health`: database and model status
- `POST /api/corrections`: upload audio or submit an example
- `POST /api/corrections/{id}/rerun`: rerun SS + GL with edited ASR text
- `GET /api/corrections`: history
- `GET /api/metrics/summary`: training metrics
- `GET /api/examples`: curated examples
