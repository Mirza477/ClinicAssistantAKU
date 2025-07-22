import os
import uuid
import pdfplumber
from tabulate import tabulate

from config.settings import PDF_DIR, CLINICAL_INSTRUCTIONS_DIR
from .cosmos_db import upsert_policy_section, get_cosmos_container
from .embeddings import generate_embedding
from src.ai_document_extractor import extract_and_embed



def extract_text_and_tables(pdf_path: str):
    """
    Extract text and tables from a single PDF file.
    """
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            sections.append({
                "text": text.strip(),
                "tables": tables
            })
    return sections



def process_clinical_instructions():
    container = get_cosmos_container()
    for fname in os.listdir(CLINICAL_INSTRUCTIONS_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        # skip if already ingested
        seen = list(container.query_items(
            query="SELECT c.id FROM c WHERE c.document_name=@doc",
            parameters=[{"name":"@doc","value": fname}],
            enable_cross_partition_query=True
        ))
        if seen:
            print(f"⏭️  {fname} already ingested.")
            continue

        path = os.path.join(CLINICAL_INSTRUCTIONS_DIR, fname)
        print(f"📄 Ingesting clinical doc: {fname}")
        extract_and_embed(path, fname)


def process_pdfs():
    """
    1) For each PDF in PDF_DIR, check if we've already ingested it (by document_name)
    2) If not, extract text & tables, embed, and upsert into Cosmos DB
    """
    if not os.path.isdir(PDF_DIR):
        print(f"PDF directory does not exist: {PDF_DIR}")
        return

    # Get a handle to your Cosmos container
    container = get_cosmos_container()

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    for pdf_file in pdf_files:
        doc_name = pdf_file  # or os.path.basename(pdf_path)
        # 1) Check for existing chunks of this document
        query = "SELECT TOP 1 c.id FROM c WHERE c.document_name=@doc"
        params = [{"name": "@doc", "value": doc_name}]
        existing = list(container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        if existing:
            print(f"Skipping {doc_name} — already ingested.")
            continue

        # 2) Not seen before, so process it
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        sections = extract_text_and_tables(pdf_path)

        # Generate a unique document_id for this document
        document_id = str(uuid.uuid4())
        print(f"Processing {doc_name} with document_id: {document_id}")

        for idx, sec in enumerate(sections, start=1):
            content = sec["text"]
            if sec["tables"]:
                for table in sec["tables"]:
                    content += "\n\nTable:\n" + tabulate(table, tablefmt="pipe")

            embedding = generate_embedding(content)
            if embedding is None:
                print(f"  • Skipping Section {idx}: embedding failed.")
                continue

            item = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,  # Add document_id to each section
                "document_name": doc_name,
                "section": f"Section {idx}",
                "content": content,
                "vector": embedding
            }

            try:
                upsert_policy_section(item)
                print(f"  ✓ Upserted Section {idx} of {doc_name}")
            except Exception as e:
                print(f"  ✗ Failed to upsert Section {idx}: {e}")


if __name__ == "__main__":
    process_pdfs()
    process_clinical_instructions()
