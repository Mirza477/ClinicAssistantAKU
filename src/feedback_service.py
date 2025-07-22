from azure.cosmos import CosmosClient, PartitionKey
from uuid import uuid4
from typing import Optional, List, Dict, Any

from pydantic import BaseModel



from config.settings import COSMOS_FEEDBACK_KEY,COSMOS_FEEDBACK_URI

DATABASE_NAME = "chatdb"

CONTAINER_NAME = "feedback"

client = CosmosClient(COSMOS_FEEDBACK_URI, credential = COSMOS_FEEDBACK_KEY)

database = client.create_database_if_not_exists(id=DATABASE_NAME)

container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/session_id"),
    offer_throughput=400
)

# Feedback model
class FeedbackIn(BaseModel):
    # session_id: Optional[str]
    question: str
    answer: str
    rating: int            # e.g. 1 = thumbs-up, 0 = thumbs-down
    comment: Optional[str] = None

def add_feedback(fb: FeedbackIn) -> str:
    """
    Insert a new feedback document into Cosmos and return its new id.
    """
    doc = fb.dict()
    doc["id"] = str(uuid4())
    # Optionally add timestamp here if you like:
    # doc["timestamp"] = datetime.utcnow().isoformat()
    container.create_item(body=doc)
    return doc["id"]

def get_all_feedback() -> List[Dict[str, Any]]:
    """
    Query and return all feedback documents, newest first.
    """
    query = "SELECT * FROM c ORDER BY c._ts DESC"
    return list(container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))