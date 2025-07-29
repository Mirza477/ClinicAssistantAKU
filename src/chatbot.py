# # src/chatbot.py
# import openai
# from config.settings import OPENAI_COMPLETIONS_DEPLOYMENT

# # from src.cosmos_db import query_vector_search

# from src.retriever import hybrid_retrieve

# from src.embeddings import generate_embedding

# # Global conversation history for the session.
# conversation_history = []

# def summarize_history(history):
#     """
#     Summarizes earlier parts of the conversation succinctly.
#     """
#     prompt = "Summarize the following conversation succinctly, capturing only key points:\n"
#     for msg in history:
#         prompt += f"{msg['role']}: {msg['content']}\n"
#     try:
#         response = openai.ChatCompletion.create(
#             engine=OPENAI_COMPLETIONS_DEPLOYMENT,
#             messages=[{"role": "system", "content": prompt}],
#             temperature=0.1,
#             max_tokens=400
#         )
#         summary = response["choices"][0]["message"]["content"].strip()
#         print("Summary generated:", summary, flush=True)
#         return summary
#     except Exception as e:
#         print("Error in summarize_history:", e, flush=True)
#         return ""

# def generate_response(user_query: str):
#     global conversation_history
#     print("generate_response called with:", user_query, flush=True)

#     # Append current user query to conversation history.
#     conversation_history.append({"role": "user", "content": user_query})

#     # 1) Generate embedding for the **current** query
#     try:
#         query_embedding = generate_embedding(user_query)
#         print("Embedding generated", flush=True)
#     except Exception as e:
#         error_msg = f"Error generating embedding: {e}"
#         print(error_msg, flush=True)
#         conversation_history.append({"role": "assistant", "content": error_msg})
#         return error_msg

#     # 2) Retrieve top‑k relevant chunks from Cosmos
#     try:
#         # docs = query_vector_search(query_embedding, top_k=10)
        
#         # Hybrid BM25 + vector retrieval
#         docs = hybrid_retrieve(user_query)
        
#         print("Relevant docs:", docs, flush=True)
        
#     except Exception as e:
#         error_msg = f"Error querying Cosmos DB: {e}"
#         print(error_msg, flush=True)
#         conversation_history.append({"role": "assistant", "content": error_msg})
#         return error_msg

#     # 3) Build system prompt
#     system_prompt = """
# You are an AKU Employee Assistant designed to answer questions based solely on the content of the company’s policy documents.
# Your role is to interpret the user’s question, locate the most relevant policy excerpt, and respond clearly and concisely.

# Please follow these instructions when providing your response:

# 1. **Use the policy documents**: Always search the provided policy documents for the answer. Extract the most pertinent information—sections, clauses, dates—as needed to give a precise reply.
# 2. **Match intent, not just keywords**: If the user’s phrasing doesn’t exactly match the document, interpret the intent (e.g. “external suppliers” ↔ “third‑party vendors”) and find the relevant policy text.
# 3. **Leverage conversation history**: If the user asks follow‑up questions, refer to earlier messages to maintain context and coherence.
# 4. **Be concise and structured**:
#    - If multiple steps or bullet points help clarity (e.g., “To request access…”), format your answer as a numbered list or bullets.
#    - Otherwise, keep answers to 2–3 sentences.
# 5. **Ask for clarification when needed**: If the question is vague or missing critical details, prompt the user for more information before attempting to answer.
# 6. **Out‑of‑scope handling**: If no relevant information exists in the documents, respond:
#    “I’m sorry, I couldn’t locate that information in the policy documents. Could you please clarify or contact the appropriate department?”
# 7. **No external knowledge**: Do not draw on any sources beyond the provided policy documents.
# 8. **Maintain a professional tone**: Always be polite, formal, and focused on policy—never make personal remarks or use informal language.
# """

#     messages = [{"role": "system", "content": system_prompt}]

#     # 4) Inject each retrieved document chunk (up to 1000 chars)
#     if docs:
#         for doc in docs:
#             excerpt = doc.get("content", "")[:2000]
#             doc_context = (
#                 f"Document: {doc.get('document_name', 'N/A')}, "
#                 f"Section: {doc.get('section', 'N/A')}.\n"
#                 f"Excerpt:\n{excerpt}"
#             )
#             messages.append({"role": "system", "content": doc_context})
#     else:
#         messages.append({
#             "role": "system",
#             "content": "No relevant documents found."
#         })

#     # 5) (Optional) include a one‐sentence summary of prior chat if available
#     if len(conversation_history) > 2:
#         summary = summarize_history(conversation_history[:-1])
#         messages.append({
#             "role": "system",
#             "content": f"Conversation so far (summarized): {summary}"
#         })

#     # 6) Always append the user’s current question _last_
#     messages.append({"role": "user", "content": user_query})

#     # 7) Call the OpenAI Chat API
#     try:
#         response = openai.ChatCompletion.create(
#             engine=OPENAI_COMPLETIONS_DEPLOYMENT,
#             messages=messages,
#             temperature=0.1,
#             max_tokens=450
#         )
#         answer = response["choices"][0]["message"]["content"].strip()
#         print("Answer received:", answer, flush=True)
#         print("Using GPT model:", OPENAI_COMPLETIONS_DEPLOYMENT)
#     except Exception as e:
#         answer = f"Error in ChatCompletion: {e}"
#         print(answer, flush=True)

#     # 8) Save and return
#     conversation_history.append({"role": "assistant", "content": answer})
#     return {"response": answer, "results": []}



# src/chatbot.py
import tiktoken
import openai
from config.settings import OPENAI_COMPLETIONS_DEPLOYMENT

# from src.cosmos_db import query_vector_search

from src.retriever import hybrid_retrieve

from src.embeddings import generate_embedding

MODEL_NAME = OPENAI_COMPLETIONS_DEPLOYMENT  
ENCODING = tiktoken.encoding_for_model(MODEL_NAME)

def rewrite_query(followup: str, history: list[dict]) -> str:
    """
    Turn a follow-up into a full question using the last user turn in history.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a question-rewriting assistant. "
                "Given the last user question and this follow-up, rewrite it into a complete, standalone question."
            )
        },
        {
            "role": "user",
            "content": (
                f"Last question: {history[-1]['content']}\n"
                f"Follow-up: {followup}"
            )
        }
    ]
    resp = openai.ChatCompletion.create(
        engine=OPENAI_COMPLETIONS_DEPLOYMENT,
        messages=prompt,
        temperature=0.0,
        max_tokens=64
    )
    return resp.choices[0].message.content.strip()



# Global conversation history for the session.
conversation_history = []

def summarize_history(history):
    """
    Summarizes earlier parts of the conversation succinctly.
    """
    prompt = "Summarize the following conversation succinctly, capturing only key points:\n"
    for msg in history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        summary = response["choices"][0]["message"]["content"].strip()
        # print("Summary generated:", summary, flush=True)
        return summary
    except Exception as e:
        print("Error in summarize_history:", e, flush=True)
        return ""


def generate_response(user_query: str, history: list[dict]) -> dict:
    
    
    global conversation_history
    # print("generate_response called with:", user_query, flush=True)

    conversation_history = history
    # Append current user query to conversation history.
    # conversation_history.append({"role": "user", "content": user_query})

    # 1) Generate embedding for the **current** query
    # try:
        # query_embedding = generate_embedding(user_query)
        # print("Embedding generated", flush=True)
    
    # except Exception as e:
    #     error_msg = f"Error generating embedding: {e}"
    #     print(error_msg, flush=True)
    #     # conversation_history.append({"role": "assistant", "content": error_msg})
    #     return error_msg


     # 1) If this looks like a follow-up (very short), rewrite it
    # tokens = user_query.strip().split()
    # if len(tokens) < 3 and history:
    #     # history[-1] must exist; we assume it’s the last user turn
    #     try:
    #         rewritten = rewrite_query(user_query, history)
    #         # swap in the rewritten question for retrieval & answering
    #         user_query = rewritten
    #         print(f"[rewrite] '{tokens}' → '{rewritten}'", flush=True)
    #     except Exception:
    #         # if rewriting fails, fall back to original user_query
    #         pass

    # print("This is updated user query-> ",user_query)

    # 2) Retrieve top‑k relevant chunks from Cosmos
    try:
        # docs = query_vector_search(query_embedding, top_k=10)
        
        # Hybrid BM25 + vector retrieval
        docs = hybrid_retrieve(user_query)
        
        print("Relevant policy names:", flush=True)
        for doc in docs:
            print("-", doc.get("document_name", "N/A"), flush=True)
        
    except Exception as e:
        error_msg = f"Error querying Cosmos DB: {e}"
        print(error_msg, flush=True)
        # conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

    # 3) Build system prompt
    system_prompt = """
    
    You are ClinicianAssistant, a professional and focused clinical chatbot developed for The Aga Khan University Hospital. You assist AKU doctors, nurses, and other licensed healthcare professionals by providing accurate information directly and exclusively from the AKU Manual of Clinical Practice Guidelines.

Your responses must be strictly based on the content of the provided clinical guidelines. Do not interpret, guess, summarize beyond what is written, or respond using your own knowledge or reasoning. Copy and present relevant information exactly as it appears in the clinical guideline documents. Your tone should remain clear, neutral, and clinically professional.

**Verbatim quoting only**: Answer *exclusively* from the retrieved excerpts.
- Whenever a recommendation includes a label such as “[Strong recommendation]”, “[Weak recommendation]”, or similar, you MUST include it exactly as written and immediately following the statement, without any changes to the label’s wording or punctuation.
- Do NOT add, infer, or modify any recommendation label, and do NOT omit it if it is present in the source.
- Do NOT hallucinate, paraphrase beyond the excerpt, or add any extra content.

If a user asks a question that is not relevant to clinical guidance or clinical practice, politely inform them that you can only assist with clinical guidelines.

If a user asks for details not found in the Manual of Clinical Practice Guidelines, politely say: “I’m sorry, I do not have that information. Please refer to the official clinical documents for more details.” Do not make suggestions, provide assumptions, or create additional content.

If a clinical recommendation includes a note such as “refer to specialist” or similar wording, you must include that referral instruction at the end of your response, exactly as stated in the guideline. If no such referral is mentioned, do not suggest or imply it yourself.

Use bullet points or tables where appropriate to clearly present steps, criteria, decision pathways, or treatment protocols, ensuring clinical clarity and ease of use.

Do not respond to personal, academic, creative, or general knowledge queries. Never provide medical advice outside of the documented guidelines. Do not explain or translate guideline content unless the user specifically asks for an explanation or translation.

Your sole purpose is to serve as a structured access tool for the AKU Manual of Clinical Practice Guidelines. Always prioritize safety, accuracy, and strict adherence to the official documentation. If a user says thank you, simply acknowledge it with a brief professional response.

Don't tell when it was the last time your knowledge was updated. Just tell I am updated on AKU Manual of Clinical Practice Guidelines.
"""

    # system_prompt = """ 
    # You are AKU Clinician Assistant, a professional and focused clinical chatbot developed for The Aga Khan University Hospital. You assist AKU doctors, nurses, and other licensed healthcare professionals by providing accurate information directly and exclusively from the AKU Manual of Clinical Practice Guidelines.
    # Speak and respond like a helpful clinical assistant. Use a clear, concise, and professional tone at all times.
    # Your responses must be strictly based on the content of the provided clinical guidelines. Do not interpret, guess, summarize beyond what is written, or respond using your own knowledge or reasoning. When you provide information, you must copy and present the exact recommendation text as it appears in the clinical guideline documents, including punctuation and phrasing. Do not reword, paraphrase, or change the meaning. 
    # When responding, always consider the entire conversation history to understand the context of follow up questions. Merge the context from previous messages with the current user query. Do not simply repeat a previous answer unless it directly addresses the new question. Instead, build upon prior context and provide additional or relevant details from the guidelines. 
    # If the user provides a clinical scenario rather than a direct question, carefully analyze the scenario to identify the key clinical elements (e.g., patient condition, setting, procedure). Retrieve and present only the exact matching recommendation text from the AKU Manual of Clinical Practice Guidelines that applies to that scenario. Do not rephrase or add interpretation. If no relevant content is found, respond: “I’m sorry, I do not have that information. Please refer to the official clinical documents for more details.” 
    # If a clinical recommendation includes a note such as “refer to specialist” or similar wording, you must include that referral instruction at the end of your response, exactly as stated in the guideline. If no such referral is mentioned, do not suggest or imply it yourself. 
    # If the guideline explicitly mentions a level of recommendation (e.g., "[Strong recommendation]" or "[Strong recommendation, low level of evidence]" or "[High evidence]" or "[Good practice point]" or similar), you must include it in your response exactly as written. 
    # Use bullet points or tables where appropriate to clearly present steps, criteria, decision pathways, or treatment protocols, ensuring clinical clarity and ease of use. 
    # Do not respond to personal, academic, creative, or general knowledge queries. Never provide medical advice outside of the documented guidelines. Do not explain or translate guideline content unless the user specifically asks for an explanation or translation. 
    # Your sole purpose is to serve as a structured access tool for the AKU Manual of Clinical Practice Guidelines. Always prioritize safety, accuracy, and strict adherence to the official documentation. 
    # If a user says thank you, simply acknowledge it with a brief professional response. Don't tell when it was the last time your knowledge was updated. Just tell: I am updated on AKU Manual of Clinical Practice Guidelines. """


    
    # 3) Start assembling the messages payload
    messages: list[dict] = [
        {"role": "system", "content": system_prompt.strip()}
    ]

     # 2) **Replay the last N turns** (user + assistant)
    for turn in history:
        messages.append({
          "role": turn["role"],       # "user" or "assistant"
          "content": turn["content"]
        })

    # 4) Inject each retrieved document chunk (up to 1000 chars)
    if docs:
        for doc in docs:
            excerpt = doc.get("content", "")
            doc_context = (
                f"Document: {doc.get('document_name', 'N/A')}, "
                f"Section: {doc.get('section', 'N/A')}.\n"
                f"Excerpt:\n{excerpt}"
            )
            messages.append({"role": "system", "content": doc_context})
    else:
        messages.append({
            "role": "system",
            "content": "No relevant documents found."
        })
    # print("history-->", history)


    # 6) Always append the user’s current question _last_
    messages.append({"role": "user", "content": user_query})

    prompt_tokens = 0
    for m in messages:
        prompt_tokens += len(ENCODING.encode(m["content"]))
    print(f"[Tokens] prompt_tokens={prompt_tokens}")

    # 7) Call the OpenAI Chat API

    # print("summary here---->",summary)

    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            # max_tokens=700,
            top_p=1.0
        )
        answer = response["choices"][0]["message"]["content"].strip()
        # print("Answer received:", answer, flush=True)
        # print("Using GPT model:", OPENAI_COMPLETIONS_DEPLOYMENT)
    except Exception as e:
        answer = f"Error in ChatCompletion: {e}"
        print(answer, flush=True)

    # 8) Save and return
    # conversation_history.append({"role": "assistant", "content": answer})
    return {"response": answer, "results": []}  
