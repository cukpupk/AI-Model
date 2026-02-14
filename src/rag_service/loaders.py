import json
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def load_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ""

    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".json":
        raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(raw, ensure_ascii=False, indent=2)

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    return ""


def collect_documents(source_dir: Path, glob_pattern: str = "**/*") -> list[tuple[Path, str]]:
    docs: list[tuple[Path, str]] = []
    for path in source_dir.glob(glob_pattern):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = load_text_from_file(path).strip()
        if text:
            docs.append((path, text))
    return docs
