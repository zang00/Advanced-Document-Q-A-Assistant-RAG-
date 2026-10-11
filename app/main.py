import logging
from threading import Lock

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.rag import RagService

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


def create_app(settings: Settings | None = None, service: RagService | None = None) -> FastAPI:
    settings = settings or Settings()
    api = FastAPI(title="Document Q&A Prototype", version="0.1.0")
    api.state.rag = service
    initialization_lock = Lock()

    def get_service() -> RagService:
        # Lazy initialization lets /health work before an API key is configured.
        with initialization_lock:
            if api.state.rag is None:
                if not settings.openai_api_key.get_secret_value():
                    raise HTTPException(503, "Set OPENAI_API_KEY in .env")
                try:
                    api.state.rag = RagService(settings)
                except Exception as exc:
                    logger.exception("Cannot initialize RAG")
                    raise HTTPException(503, "Cannot initialize document storage") from exc
        return api.state.rag

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "api_key_configured": bool(
            settings.openai_api_key.get_secret_value()
        )}

    @api.post("/upload")
    async def upload(file: UploadFile = File(...), rag: RagService = Depends(get_service)) -> dict:
        # Read at most the configured limit, then process synchronously off the event loop.
        try:
            content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(content) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "File exceeds the upload size limit")
            return await run_in_threadpool(rag.upload, file.filename or "", content)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Upload failed")
            raise HTTPException(502, "Document indexing failed; check the API key and retry") from exc
        finally:
            await file.close()

    @api.post("/chat")
    def chat(request: ChatRequest, rag: RagService = Depends(get_service)) -> dict:
        try:
            return rag.chat(request.question, request.session_id)
        except Exception as exc:
            logger.exception("Chat failed")
            raise HTTPException(502, "Could not generate an answer; please retry") from exc

    @api.get("/documents")
    def documents(rag: RagService = Depends(get_service)) -> dict:
        try:
            return {"documents": rag.documents()}
        except Exception as exc:
            logger.exception("Document listing failed")
            raise HTTPException(503, "Cannot read document storage") from exc

    return api


app = create_app()
