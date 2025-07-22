# src/conversation_service.py

from uuid import uuid4
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from azure.cosmos import CosmosClient, PartitionKey

# Adjust these to match your Settings
from config.settings import COSMOS_FEEDBACK_URI, COSMOS_FEEDBACK_KEY

#––– Cosmos setup –––#
DATABASE_NAME  = "chatdb"
CONTAINER_NAME = "chathistory"

client = CosmosClient(COSMOS_FEEDBACK_URI, credential=COSMOS_FEEDBACK_KEY)
db     = client.create_database_if_not_exists(id=DATABASE_NAME)
container = db.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/session_id"),
    offer_throughput=400
)

# ————————————————
# 2) define the record model
# ————————————————
class ConversationRecord(BaseModel):
    session_id: str
    question:   str
    answer:     str
    timestamp:  Optional[str] = None  # will auto-fill
    id:         Optional[str] = None  # will auto-fill

# ————————————————
# 3) write helper to insert one turn
# ————————————————
def add_record(rec: ConversationRecord) -> str:
    item = rec.dict()
    if not item.get("id"):
        item["id"] = str(uuid4())
    if not item.get("timestamp"):
        item["timestamp"] = datetime.utcnow().isoformat()
    container.create_item(body=item)
    return item["id"]