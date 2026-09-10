"""
Text chunking utilities.
Splits long law texts into smaller overlapping pieces so they can be
embedded and retrieved effectively by the RAG system (Person 3's module).
This module only exports functions - it doesn't run a pipeline by itself.
"""

# Default sizes are in WORDS, not characters - this avoids cutting
# Arabic words in half, which character-based slicing would do.
DEFAULT_CHUNK_SIZE = 200   # words per chunk
DEFAULT_OVERLAP = 40       # words shared between consecutive chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> list:
    """
    Splits text into overlapping chunks, measured in whole words.

    Example: chunk_size=200, overlap=40 means each chunk has 200 words,
    and the next chunk starts 160 words after the previous one started
    (so the last 40 words of a chunk reappear at the start of the next).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks
