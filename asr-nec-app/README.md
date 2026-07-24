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
Compose uses host networking because the 240 server's predefined Docker
address pools are fully allocated.

## API

- `GET /api/health`: database and model status
- `POST /api/corrections`: upload audio or submit an example
- `POST /api/corrections/{id}/rerun`: rerun SS + GL with edited ASR text
- `GET /api/corrections`: history
- `GET /api/metrics/summary`: training metrics
- `GET /api/examples`: curated examples
