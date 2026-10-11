from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def parse_document(filename: str, content: bytes) -> list[Document]:
    # Keep the display name only; uploaded filenames never become filesystem paths.
    name = Path(filename.replace("\\", "/")).name
    extension = Path(name).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported")
    if not content:
        raise ValueError("The file is empty")

    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT files must use UTF-8 encoding") from exc
        pages = [Document(page_content=text, metadata={"document": name})]
    else:
        # Parse each page separately so chunks retain physical PDF page numbers.
        try:
            if not content.startswith(b"%PDF-"):
                raise ValueError("Invalid PDF header")
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported")
            pages = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"document": name, "page": number},
                )
                for number, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError("Cannot read this PDF; it may be corrupt or encrypted") from exc

    pages = [page for page in pages if page.page_content.strip()]
    if not pages:
        raise ValueError("No readable text found; scanned PDFs require OCR")
    return pages
