# src/embeddings.py
import logging
import openai
import tiktoken
from config.settings import OPENAI_ENDPOINT, OPENAI_KEY, OPENAI_EMBEDDINGS_DEPLOYMENT
from tenacity import retry, wait_random_exponential, stop_after_attempt


# prepare tokenizer for your embedding model
ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 1000

openai.api_type = "azure"
openai.api_base = OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = OPENAI_KEY

@retry(wait=wait_random_exponential(min=2, max=120), stop=stop_after_attempt(6))
def generate_embedding(last_message: str):
    # try:
    #     response = openai.Embedding.create(
    #         engine=OPENAI_EMBEDDINGS_DEPLOYMENT,
    #         input=[last_message]
    #     )
    #     embedding = response["data"][0]["embedding"]
    #     return embedding
    # except Exception as e:
    #     logging.error("Error generating embedding: %s", e)
    #     raise


        # split into ≤MAX_TOKENS if needed
    tokens = ENC.encode(last_message)
    if len(tokens) <= MAX_TOKENS:
        parts = [last_message]
    else:
        parts = [
            ENC.decode(tokens[i : i + MAX_TOKENS])
            for i in range(0, len(tokens), MAX_TOKENS)
        ]
    embeddings = []
    for part in parts:
        try:
            resp = openai.Embedding.create(
                engine=OPENAI_EMBEDDINGS_DEPLOYMENT,
                input=[part]
            )
            embeddings.append(resp["data"][0]["embedding"])
        except Exception as e:
            logging.error("Error generating embedding for a slice: %s", e)
            raise

    # if multiple slices, average the vectors element-wise
    if len(embeddings) == 1:
        return embeddings[0]
    avg = [
        sum(vals) / len(vals)
        for vals in zip(*embeddings)
    ]
    return avg
