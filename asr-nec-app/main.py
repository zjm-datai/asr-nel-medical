from __future__ import annotations

import uvicorn

from configs.base import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
