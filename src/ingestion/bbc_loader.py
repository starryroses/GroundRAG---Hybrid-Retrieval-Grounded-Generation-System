from pathlib import Path
from src.models import Document

BBC_CATEGORIES = {
    "business",
    "entertainment",
    "politics",
    "sport",
    "tech",
}

def load_bbc_corpus(data_path: str) -> list[Document]:
    """
    Load the BBC News 2225-document corpus.
    Expected structure:
    data_path/
        business/
        entertainment/
        politics/
        sport/
        tech/
    Each article has:
        first non-empty line -> title
        remaining text       -> article body
    """
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"BBC dataset directory does not exist: {root}")
    
    documents = []
    for category in sorted(BBC_CATEGORIES):
        category_path = root / category
        if not category_path.exists():
            raise FileNotFoundError(f"Missing BBC category directory: {category_path}")
        for file_path in sorted(category_path.glob("*.txt")):
            raw_text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
            lines = [
                line.strip()
                for line in raw_text.splitlines()
                if line.strip()
            ]
            if not lines:
                continue
            title = lines[0]
            article_text = " ".join(lines[1:]).strip()
            if not article_text:
                continue
            document_id = f"{category}/{file_path.stem}"
            document = Document(
                document_id=document_id,
                document_name=file_path.name,
                title=title,
                text=article_text,
                category=category,
                source_path=str(file_path),
            )
            documents.append(document)

    if len(documents) != 2225:
        raise ValueError(
            f"Expected 2225 documents, "
            f"but loaded {len(documents)}."
        )
    
    return documents