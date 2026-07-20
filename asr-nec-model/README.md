# ASR NEC Model

Project scaffold for the ASR named entity correction model.

## Layout

- `src/asr_nec_model/models`: model modules
- `src/asr_nec_model/data`: datasets and batch helpers
- `src/asr_nec_model/training`: training loops
- `src/asr_nec_model/inference`: retrieval and correction pipeline
- `src/asr_nec_model/audio`: feature extraction
- `src/asr_nec_model/utils`: shared helpers

## Quick start

```bash
uv sync
uv run python -m asr_nec_model
```

