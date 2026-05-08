from typing import List


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20):

    if not text or not isinstance(text, str):
        return []

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = words[i : i + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks
