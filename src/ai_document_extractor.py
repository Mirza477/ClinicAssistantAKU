# src/ai_document_extractor.py

import uuid
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from .embeddings import generate_embedding
from .cosmos_db import upsert_policy_section, get_cosmos_container
from tabulate import tabulate
from config.settings import FORM_RECOGNIZER_ENDPOINT, FORM_RECOGNIZER_KEY

def extract_and_embed(pdf_path: str, doc_name: str):
    # 1) Initialize client
    client = DocumentIntelligenceClient(
        FORM_RECOGNIZER_ENDPOINT,
        AzureKeyCredential(FORM_RECOGNIZER_KEY)
    )

    # 2) Analyze with layout model
    with open(pdf_path, "rb") as stream:
        poller = client.begin_analyze_document("prebuilt-layout", stream)
        result = poller.result()

    # 3) Chunk by headings + paragraphs
    chunks = []
    current = {"title": None, "content": ""}

    for para in result.paragraphs or []:
        text = para.content.strip()
        is_heading = (
            getattr(para, "role", None) == "sectionHeading"
            or text.isupper()
        )
        if is_heading:
            if current["title"]:
                chunks.append(current)
            current = {"title": text, "content": ""}
        else:
            current["content"] += text + "\n\n"

    # append last chunk
    if current["title"]:
        chunks.append(current)

    # 4) Attach **all** tables to the last chunk
    for tbl in result.tables or []:
        # build a matrix from row_count, column_count, and tbl.cells
        matrix = [["" for _ in range(tbl.column_count)] for _ in range(tbl.row_count)]
        for cell in tbl.cells:
            matrix[cell.row_index][cell.column_index] = cell.content
        md = "\n".join("| " + " | ".join(row) + " |" for row in matrix)
        if chunks:
            chunks[-1]["content"] += "\n\nTable:\n" + md + "\n\n"

    # 5) Flag selected checkboxes/radios (red dots)
    # selection_marks live under each page
    for page in result.pages or []:
        for mark in page.selection_marks or []:
            if mark.state == "selected" and chunks:
                chunks[-1]["content"] += "\n[⚠️] Selection marked here\n\n"

    # 6) Embed & upsert
    container = get_cosmos_container()
    document_id = str(uuid.uuid4())

    MAX_CHUNK_SIZE = 1000  # characters per embedding call

    for chunk in chunks:
        text = chunk["content"]
        # simple character-based slicing; you can swap in a token-based splitter if you like
        parts = [
            text[i : i + MAX_CHUNK_SIZE]
            for i in range(0, len(text), MAX_CHUNK_SIZE)
        ]

        for idx, part in enumerate(parts, start=1):
            section_name = chunk["title"]
            if len(parts) > 1:
                section_name = f"{section_name} (part {idx})"

            try:
                emb = generate_embedding(part)
            except Exception as e:
                # log & skip this slice, but keep going
                print(f"  ⚠️ Embedding failed for {section_name}: {e}")
                continue

            item = {
                "id":            str(uuid.uuid4()),
                "document_id":   document_id,
                "document_name": doc_name,
                "section":       section_name,
                "content":       part,
                "vector":        emb
           }
            upsert_policy_section(item)

    print(f"✓ Ingested {len(chunks)} chunks from {doc_name}")
