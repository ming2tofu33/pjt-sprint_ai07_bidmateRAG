"""FastAPI app entry point for BidMate web_api adapter."""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bidmate_rag.evaluation.dataset import find_latest_metadata_path
from bidmate_rag.storage.metadata_store import MetadataStore
from bidmate_rag.web_api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = find_latest_metadata_path()
    if path.exists():
        app.state.metadata_store = MetadataStore.from_parquet(path)
    else:
        import pandas as pd
        app.state.metadata_store = MetadataStore(pd.DataFrame())
    yield


app = FastAPI(title="BidMate Web API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
