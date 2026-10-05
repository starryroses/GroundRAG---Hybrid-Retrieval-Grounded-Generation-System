from src.models import Document, Chunk

TARGET_CHARS = 700
OVERLAP_CHARS = 100

def _create_chunk(
    document: Document,
    sentences: list[str],
    chunk_index: int,
) -> Chunk:

    text = " ".join(sentences).strip()

    return Chunk(
        chunk_id=f"{document.document_id}::chunk_{chunk_index}",
        document_id=document.document_id,
        document_name=document.document_name,
        title=document.title,
        category=document.category,
        chunk_index=chunk_index,
        text=text,
        source_path=document.source_path,
    )

def chunk_document(
    document: Document,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:

    if not document.sentences:
        raise ValueError(
            f"Document {document.document_id} has no sentences."
        )

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_index = 0

    for sentence in document.sentences:
        sentence_length = len(sentence)

        # If adding this sentence exceeds the target, finalize the current chunk.
        if (current_sentences and current_length + sentence_length > target_chars):

            chunks.append(
                _create_chunk(
                    document,
                    current_sentences,
                    chunk_index,
                )
            )

            chunk_index += 1

            # Build overlap from the end of the previous chunk.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):
                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence)

                # Always keep at least one complete sentence.
                if overlap_length >= overlap_chars:
                    break

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length

    # Add the final chunk.
    if current_sentences:
        chunks.append(
            _create_chunk(
                document,
                current_sentences,
                chunk_index,
            )
        )

    return chunks