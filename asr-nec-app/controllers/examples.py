from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from configs.base import Settings, get_settings
from models.schemas import ExampleListResponse, ExampleRead
from services import examples_service

router = APIRouter(prefix="/api/examples", tags=["examples"])


@router.get("", response_model=ExampleListResponse)
def list_examples(
    settings: Settings = Depends(get_settings),
) -> ExampleListResponse:
    rows = examples_service.list_examples(settings)
    return ExampleListResponse(
        examples=[
            ExampleRead(**{key: value for key, value in row.items() if key != "audio_path"})
            for row in rows
        ]
    )


@router.get("/{example_id}/audio")
def get_example_audio(
    example_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    example = examples_service.get_example(settings, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="example not found")
    audio_path = (settings.examples_audio_dir / example["audio_path"]).resolve()
    root = settings.examples_audio_dir.resolve()
    if root not in audio_path.parents or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="example audio not found")
    return FileResponse(audio_path)
