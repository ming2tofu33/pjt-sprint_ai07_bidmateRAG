"""FastAPI app entry point for BidMate web_api adapter."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bidmate_rag.evaluation.dataset import find_latest_metadata_path
from bidmate_rag.storage.metadata_store import MetadataStore
from bidmate_rag.web_api.routes import router

_WEB_CONFIG_PATH = Path("configs/web.yaml")
_WEB_CONFIG_DEFAULTS = {
    "provider_config": "openai_gpt5mini",
    "chunking_config": None,
    "top_k": 5,
    "max_context_chars": 8000,
}


def _load_web_config() -> dict:
    """configs/web.yaml 로 부터 웹 UI 기본 조합을 읽는다.

    파일이 없으면 하드코딩 디폴트로 폴백 (테스트·개발 환경 대비).
    """
    if not _WEB_CONFIG_PATH.exists():
        return dict(_WEB_CONFIG_DEFAULTS)
    loaded = yaml.safe_load(_WEB_CONFIG_PATH.read_text()) or {}
    return {**_WEB_CONFIG_DEFAULTS, **loaded}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = find_latest_metadata_path()
    if path.exists():
        app.state.metadata_store = MetadataStore.from_parquet(path)
    else:
        import pandas as pd
        app.state.metadata_store = MetadataStore(pd.DataFrame())
    app.state.web_config = _load_web_config()
    yield


app = FastAPI(title="BidMate Web API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
