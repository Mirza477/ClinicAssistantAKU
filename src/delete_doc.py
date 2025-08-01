# src/delete_doc.py
from config.settings     import COSMOS_CONTAINER
from src.cosmos_db       import get_cosmos_container
from azure.cosmos         import PartitionKey

def delete_document(doc_name: str):
    container = get_cosmos_container()
    # fetch all items for that document
    query = "SELECT c.id, c.document_name FROM c WHERE c.document_name=@doc"
    params = [{"name":"@doc","value":doc_name}]
    items = list(container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    ))
    print(f"Found {len(items)} items for '{doc_name}' – deleting…")
    for item in items:
        # partition key is document_name
        container.delete_item(item=item["id"], partition_key=item["document_name"])
    print("Done.")

if __name__ == "__main__":
    delete_document("PostOp Anesthesia Care v1.pdf")
