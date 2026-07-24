from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from swagger_ui_bundle import swagger_ui_path  # type: ignore[import-untyped]

from configs.base import Settings


def init_app(app: FastAPI, settings: Settings) -> None:
    docs_asset_url = f"{settings.root_path}/_docs"
    app.mount(
        "/_docs",
        StaticFiles(directory=swagger_ui_path),
        name="swagger-ui-assets",
    )

    @app.get("/docs", include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=f"{settings.root_path}{app.openapi_url}",
            title=f"{settings.app_name} - Swagger UI",
            oauth2_redirect_url=f"{settings.root_path}/docs/oauth2-redirect",
            swagger_js_url=f"{docs_asset_url}/swagger-ui-bundle.js",
            swagger_css_url=f"{docs_asset_url}/swagger-ui.css",
            swagger_favicon_url=f"{docs_asset_url}/favicon-32x32.png",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    def swagger_ui_redirect() -> HTMLResponse:
        return get_swagger_ui_oauth2_redirect_html()
