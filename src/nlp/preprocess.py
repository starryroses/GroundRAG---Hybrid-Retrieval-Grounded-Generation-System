import re
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

def normalize_text(text: str) -> str:
    """
    Conservative text normalization.
    We intentionally preserve: names, numbers, technical terms, punctuation that may carry meaning.
    BBC articles contain typographic characters (‘word’, “quote”...) We normalize those into standard characters ('word', 'quote' etc)
    """
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text) #whitespace normalization
    return text.strip()


def sentence_split(text: str) -> list[str]:
    """
    Split an article into sentences while preserving the original wording.
    """
    doc = nlp(text)
    return [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]


def preprocess_document(document):
    """
    Apply NLP preprocessing to one Document.
    """
    normalized = normalize_text(document.text)
    document.normalized_text = normalized
    document.sentences = sentence_split(normalized)
    return document

def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenization specifically for sparse lexical retrieval. Unlike document normalization, this representation
    is allowed to be lowercase because it is used only for BM25 matching.
    """

    text = normalize_text(text)
    tokens = re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
    )
    return tokens

def preprocess_query(query: str) -> dict:
    """
    Prepare a user query for retrieval.
    """
    normalized = normalize_text(query)
    return {
        "original": query,
        "normalized": normalized,
        "bm25_tokens": tokenize_for_bm25(normalized),
    }