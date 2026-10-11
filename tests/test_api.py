import json
import subprocess
import sys
from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import Settings
from app.main import create_app
from app.rag import Answer, NOT_FOUND, RagService


@pytest.fixture
def setup(tmp_path):
    # Mock paid API calls, but exercise parsing and embedded Chroma for real.
    settings = Settings(_env_file=None, chroma_dir=str(tmp_path / "chroma"), max_upload_mb=1)
    llm = Mock()
    llm.invoke.return_value = Answer(found=True, answer="Wear insulated gloves.", source_ids=[0])
    embeddings = DeterministicFakeEmbedding(size=16)
    rag = RagService(settings, embeddings=embeddings, llm=llm)
    with TestClient(create_app(settings, rag)) as client:
        yield client, rag, llm, embeddings, settings


def make_pdf(*texts: str) -> bytes:
    # Build a tiny, real text PDF in memory; no binary fixture files needed.
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_txt_upload_chat_citations_and_duplicate(setup):
    client, rag, llm, _, _ = setup
    text = "At temperatures below -20 C, wear insulated gloves."
    upload = client.post("/upload", files={"file": ("../safety.txt", text.encode())})
    assert upload.status_code == 200
    assert upload.json()["document"] == "safety.txt"
    assert upload.json()["chunks"] == 1
    assert not upload.json()["duplicate"]
    assert client.post("/upload", files={"file": ("copy.txt", text.encode())}).json()["duplicate"]
    assert len(client.get("/documents").json()["documents"]) == 1
    assert client.post("/upload", files={"file": ("copy.docx", text.encode())}).status_code == 400

    # Extra model fields cannot replace real source metadata.
    llm.invoke.return_value = {
        "found": True, "answer": "Wear insulated gloves.", "source_ids": [0],
        "document": "invented.pdf", "page": 999, "snippet": "fabricated quote",
    }
    response = client.post("/chat", json={"question": "What should I wear?"})
    assert response.status_code == 200
    assert response.json()["citations"] == [{"document": "safety.txt", "page": None, "snippet": text}]
    assert len(rag.sessions[response.json()["session_id"]]) == 2


def test_pdf_page_metadata(setup):
    client, _, llm, _, _ = setup
    response = client.post("/upload", files={"file": (
        "manual.pdf", make_pdf("Wear gloves.", "Use approved equipment."), "application/pdf"
    )})
    assert response.status_code == 200
    llm.invoke.return_value = Answer(found=True, answer="Follow both instructions.", source_ids=[0, 1])
    result = client.post("/chat", json={"question": "What are the instructions?"}).json()
    assert sorted(citation["page"] for citation in result["citations"]) == [1, 2]
    assert {citation["snippet"].strip() for citation in result["citations"]} == {
        "Wear gloves.", "Use approved equipment."
    }


def test_empty_index_refuses_without_llm_call(setup):
    client, _, llm, _, _ = setup
    result = client.post("/chat", json={"question": "What is the answer?"}).json()
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("answer", [
    Answer(found=False, answer="An unsupported guess", source_ids=[0]),
    Answer(found=True, answer="A guess", source_ids=[]),
    Answer(found=True, answer="A guess", source_ids=[999]),
    Answer(found=True, answer="A guess", source_ids=[-1]),
    Answer(found=True, answer=" ", source_ids=[0]),
])
def test_missing_evidence_always_uses_exact_fallback(setup, answer):
    client, _, llm, _, _ = setup
    client.post("/upload", files={"file": ("notes.txt", b"Only gloves are discussed.")})
    llm.invoke.return_value = answer
    result = client.post("/chat", json={"question": "What is the warranty?"}).json()
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []


def test_followup_history_and_session_isolation(setup):
    client, rag, llm, _, _ = setup
    client.post("/upload", files={"file": ("notes.txt", b"Wear insulated gloves in cold weather.")})
    first = client.post("/chat", json={"question": "What gloves?", "session_id": "alice"})
    assert first.status_code == 200
    client.post("/chat", json={"question": "When should I wear them?", "session_id": "alice"})
    messages = llm.invoke.call_args.args[0]
    assert any(message.content == "What gloves?" for message in messages)
    assert len(rag.sessions["alice"]) == 4
    client.post("/chat", json={"question": "What gloves?", "session_id": "bob"})
    assert len(llm.invoke.call_args.args[0]) == 2  # System and current question only.


def test_index_persists_but_history_does_not(setup):
    client, _, llm, embeddings, settings = setup
    client.post("/upload", files={"file": ("persistent.txt", b"Wear insulated gloves.")})
    client.post("/chat", json={"question": "What gloves?", "session_id": "old"})
    reopened = RagService(settings, embeddings=embeddings, llm=llm)
    assert reopened.documents()[0]["document"] == "persistent.txt"
    assert reopened.sessions == {}
    assert reopened.chat("What gloves?")["citations"][0]["document"] == "persistent.txt"
    # A separate interpreter cannot reuse Chroma's in-process client cache.
    script = """
import json, sys
from unittest.mock import Mock
from langchain_core.embeddings import DeterministicFakeEmbedding
from app.config import Settings
from app.rag import RagService
service = RagService(Settings(_env_file=None, chroma_dir=sys.argv[1]),
                     embeddings=DeterministicFakeEmbedding(size=16), llm=Mock())
print(json.dumps(service.documents()))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, settings.chroma_dir],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert json.loads(process.stdout)[0]["document"] == "persistent.txt"


@pytest.mark.parametrize("filename,content", [
    ("notes.docx", b"unsupported"), ("notes.txt", b""),
    ("notes.txt", b" \n "), ("notes.txt", b"\xff\xfe"),
    ("broken.pdf", b"not a PDF"), ("scan.pdf", make_pdf("")),
])
def test_invalid_uploads(setup, filename, content):
    client, _, _, _, _ = setup
    assert client.post("/upload", files={"file": (filename, content)}).status_code == 400
    assert client.get("/documents").json()["documents"] == []


def test_upload_limit_and_question_validation(setup):
    client, _, _, _, _ = setup
    assert client.post("/upload", files={"file": ("large.txt", b"x" * (1024 * 1024 + 1))}).status_code == 413
    for payload in ({"question": " "}, {"question": "x" * 2001},
                    {"question": "hello", "session_id": "../bad"}):
        assert client.post("/chat", json=payload).status_code == 422


def test_provider_error_does_not_commit_history(setup):
    client, rag, llm, _, _ = setup
    client.post("/upload", files={"file": ("notes.txt", b"Wear gloves.")})
    llm.invoke.side_effect = RuntimeError("secret provider details")
    response = client.post("/chat", json={"question": "What gloves?", "session_id": "failed"})
    assert response.status_code == 502
    assert "secret" not in response.text
    assert "failed" not in rag.sessions


def test_health_without_key():
    with TestClient(create_app(Settings(_env_file=None, openai_api_key=""))) as client:
        assert client.get("/health").json() == {"status": "ok", "api_key_configured": False}
        assert client.post("/chat", json={"question": "hello"}).status_code == 503


def test_russian_text_and_deduplicated_citations(setup):
    client, _, llm, _, _ = setup
    text = "При температуре ниже -20°C используйте утеплённые перчатки."
    assert client.post("/upload", files={"file": ("safety.txt", text.encode("utf-8"))}).status_code == 200
    llm.invoke.return_value = Answer(found=True, answer="Используйте утеплённые перчатки.", source_ids=[0, 0])
    result = client.post("/chat", json={"question": "Какие перчатки нужны?"}).json()
    assert len(result["citations"]) == 1
    assert result["citations"][0]["snippet"] == text


def test_production_adapters_initialize_without_network(tmp_path):
    # Construct the real OpenAI adapters; no embedding or generation call is made.
    service = RagService(Settings(
        _env_file=None, openai_api_key="test-only-key", chroma_dir=str(tmp_path / "chroma")
    ))
    assert service.documents() == []
