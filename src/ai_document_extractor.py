import re
import uuid
import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from .cosmos_db import upsert_policy_section, get_cosmos_container
from config.settings import FORM_RECOGNIZER_ENDPOINT, FORM_RECOGNIZER_KEY

def parse_recommendation_blocks(text):
    """
    Custom parser for recommendation blocks formatted like:
    Section Heading: Table of Recommendations
    Main Heading: Management
    Subheading1: Frail, older patients
    Recommendation: ...
    Strength and level of evidence: ...
    Refer to specialist: Yes/No
    (multiple blocks in same text allowed)
    """
    # This regex will match all blocks between Section Heading and next Section Heading
    pattern = re.compile(
        r"Section Heading: Table of Recommendations\s*"
        r"Main Heading:\s*(.*?)\s*"
        r"Subheading1:\s*(.*?)\s*"
        r"Recommendation:\s*(.*?)\s*"
        r"Strength and level of evidence:\s*(.*?)\s*"
        r"Refer to specialist:\s*(.*?)\s*(?=Section Heading:|$)",
        re.DOTALL | re.IGNORECASE,
    )

    blocks = []
    for match in pattern.finditer(text):
        blocks.append({
            "section": match.group(1).strip(),
            "subsection": match.group(2).strip(),
            "recommendation": match.group(3).strip(),
            "label": match.group(4).strip(),
            "refer_specialist": match.group(5).strip().lower().startswith("y")
        })
    return blocks

def extract_and_embed(pdf_path: str, doc_name: str):
    client = DocumentIntelligenceClient(
        FORM_RECOGNIZER_ENDPOINT,
        AzureKeyCredential(FORM_RECOGNIZER_KEY)
    )
    with open(pdf_path, "rb") as stream:
        poller = client.begin_analyze_document("prebuilt-layout", stream)
        result = poller.result()

    container = get_cosmos_container()
    document_id = str(uuid.uuid4())
    all_recs = []

    # Step 1: Join all paragraphs into one big text block (for easier recommendation block matching)
    text_blocks = [para.content.strip() for para in result.paragraphs or [] if para.content.strip()]
    big_text = "\n".join(text_blocks)

    # Step 2: Extract custom recommendations blocks from text
    rec_blocks = parse_recommendation_blocks(big_text)
    all_recs.extend(rec_blocks)

    # Step 3: Remove all parsed blocks from text (so they don't get duplicated as general recs)
    for rec in rec_blocks:
        block_text = (
            f"Section Heading: Table of Recommendations\n"
            f"Main Heading: {rec['section']}\n"
            f"Subheading1: {rec['subsection']}\n"
            f"Recommendation: {rec['recommendation']}\n"
            f"Strength and level of evidence: {rec['label']}\n"
            f"Refer to specialist: {'Yes' if rec['refer_specialist'] else 'No'}"
        )
        big_text = big_text.replace(block_text, "")

    # Step 4: Now process remaining paragraphs as "normal" recommendations
    heading_pat = re.compile(r"^[A-Z][A-Z\s]{3,}$")
    subheading_pat = re.compile(r"^[A-Z][a-z\s\-]+$")
    heading = ""
    subheading = ""
    for para in big_text.split('\n'):
        text = para.strip()
        if not text or text.startswith("Section Heading: Table of Recommendations"):
            continue
        if heading_pat.fullmatch(text):
            heading = text
            subheading = ""
            continue
        if subheading_pat.fullmatch(text) and not text.isupper():
            subheading = text
            continue
        # Only treat as recommendation if not a heading or subheading
        if not heading_pat.fullmatch(text) and not subheading_pat.fullmatch(text):
            all_recs.append({
                "section": heading,
                "subsection": subheading,
                "recommendation": text,
                "label": "",
                "refer_specialist": "refer" in text.lower(),
            })

    # Step 5: Save to Cosmos DB
    for rec in all_recs:
        item = {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "document_name": doc_name,
            **rec
        }
        upsert_policy_section(item)
    print(f"✓ Ingested {len(all_recs)} recommendations from {doc_name}")

